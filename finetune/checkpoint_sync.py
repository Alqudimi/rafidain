#!/usr/bin/env python3
"""Checkpoint synchronisation: resumable training + Google Drive mirroring.

Seed-VC's upstream ``train.py`` already resumes from the newest
``DiT_epoch_*_step_*.pth`` inside its run directory — but on Google Colab that
directory lives on the ephemeral VM disk, so a recycled runtime loses hours of
training. This module fixes that with three primitives:

* :func:`latest_checkpoint` — newest rolling checkpoint in a directory.
* :func:`restore`           — pull the newest remote checkpoint back into the
  local run directory (used at training start).
* :func:`push_once` / :class:`CheckpointSync` — mirror new checkpoints (plus
  config + console log) to a remote directory (Google Drive on Colab, any
  mounted path elsewhere) while training runs, pruning old remote copies to
  respect Drive quotas.

The remote directory is generic: anything that looks like a mounted Google
Drive (``/content/drive/MyDrive``) is auto-detected and auto-mounted when
running inside Colab; ``$RAFIDAIN_GDRIVE_DIR`` overrides it anywhere.

CLI (usable standalone, e.g. from shell scripts):

    python finetune/checkpoint_sync.py watch \
        --local-dir runs/iraqi_v1 --interval 60
    python finetune/checkpoint_sync.py restore --run-name iraqi_v1
    python finetune/checkpoint_sync.py status --run-name iraqi_v1
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

CHECKPOINT_GLOB = "DiT_epoch_*_step_*.pth"
_STEP_RE = re.compile(r"step_(\d+)\.pth$")
#: Files mirrored alongside the rolling checkpoints (never pruned).
_AUX_FILES = ("ft_model.pth", "config_patched.yml", "train_console.log")

_DEFAULT_GDRIVE_MOUNT = "/content/drive/MyDrive"
_REMOTE_ROOT_NAME = "RafidainRuns"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def detect_google_drive(*, auto_mount: bool = True) -> Path | None:
    """Return the Google Drive root (``…/MyDrive``) or ``None``.

    Resolution order: ``$RAFIDAIN_GDRIVE_DIR`` > an already-mounted Colab
    Drive > mount it via ``google.colab.drive`` (Colab only, best-effort).
    """
    env = os.environ.get("RAFIDAIN_GDRIVE_DIR", "").strip()
    if env:
        path = Path(env).expanduser()
        return path if path.is_dir() else None

    candidate = Path(_DEFAULT_GDRIVE_MOUNT)
    if candidate.is_dir():
        return candidate
    if not auto_mount:
        return None

    # Colab: ask the runtime to mount Drive (prints its own permission UI).
    try:  # pragma: no cover - requires a live Colab runtime
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount("/content/drive")
    except Exception:  # noqa: BLE001 — ImportError or user refusal
        return None
    return candidate if candidate.is_dir() else None


def default_remote_root(gdrive: Path | None) -> Path | None:
    """``<gdrive>/RafidainRuns`` — one folder per run name below it."""
    return (gdrive / _REMOTE_ROOT_NAME) if gdrive else None


def run_dir_for(seed_vc_root: Path, run_name: str) -> Path:
    """Upstream run directory: ``<seed-vc>/runs/<run_name>``.

    ``train.py`` resolves ``log_dir: "./runs/"`` against its CWD (the seed-vc
    repo root), so checkpoints always land here regardless of the caller.
    """
    return seed_vc_root / "runs" / run_name


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _step_of(path: Path) -> int:
    match = _STEP_RE.search(path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(directory: Path) -> Path | None:
    """Newest ``DiT_epoch_*_step_*.pth`` in ``directory`` (by step number)."""
    if not directory.is_dir():
        return None
    candidates = [
        p for p in directory.glob(CHECKPOINT_GLOB) if _step_of(p) >= 0
    ]
    return max(candidates, key=_step_of) if candidates else None


def _same(src: Path, dst: Path) -> bool:
    try:
        return (
            src.stat().st_size == dst.stat().st_size
            and int(src.stat().st_mtime) == int(dst.stat().st_mtime)
        )
    except OSError:
        return False


def _copy_atomic(src: Path, dst: Path) -> None:
    """Copy with a temp name + rename so a killed sync never leaves halves."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def restore(local_dir: Path, remote_dir: Path | None) -> Path | None:
    """Bring the newest *remote* checkpoint back into ``local_dir``.

    Returns the checkpoint to resume from (remote-restored or already-local,
    whichever is newer), or ``None`` when there is nothing to resume from.
    Safe to call repeatedly — copies are skipped when files already match.
    """
    local_best = latest_checkpoint(local_dir)
    if remote_dir is None or not remote_dir.is_dir():
        return local_best
    remote_best = latest_checkpoint(remote_dir)
    if remote_best is None:
        return local_best

    if local_best is None or _step_of(remote_best) > _step_of(local_best):
        print(
            f"[sync] restoring checkpoint from remote: {remote_best.name} "
            f"(local was {local_best.name if local_best else '∅'})",
            flush=True,
        )
        _copy_atomic(remote_best, local_dir / remote_best.name)
        # The optimizer/scheduler state lives inside the same file; upstream
        # loads params from it, and keeping the filename lets train.py's own
        # latest-checkpoint scan pick it up automatically.
        for name in _AUX_FILES:
            src, dst = remote_dir / name, local_dir / name
            if src.is_file() and not _same(src, dst):
                _copy_atomic(src, dst)
        return local_dir / remote_best.name
    return local_best


def push_once(
    local_dir: Path,
    remote_dir: Path | None,
    *,
    keep_last: int = 3,
    include_final: bool = False,
) -> list[Path]:
    """Mirror new checkpoints ``local → remote``; prune old remote copies.

    * Only copies files whose (size, mtime) differ from the remote copy.
    * Keeps the newest ``keep_last`` checkpoints remotely; older ones are
      deleted *remotely only* (the local rolling window is upstream's job).
    * ``include_final`` also mirrors ``ft_model.pth`` (end-of-training sync).
    Returns the list of files copied this round.
    """
    if remote_dir is None or not local_dir.is_dir():
        return []
    copied: list[Path] = []

    for ckpt in sorted(local_dir.glob(CHECKPOINT_GLOB), key=_step_of):
        dst = remote_dir / ckpt.name
        if not _same(ckpt, dst):
            _copy_atomic(ckpt, dst)
            copied.append(ckpt)

    names = ["config_patched.yml", "train_console.log"]
    if include_final:
        names.append("ft_model.pth")
    for name in names:
        src = local_dir / name
        if src.is_file():
            dst = remote_dir / name
            if not _same(src, dst):
                _copy_atomic(src, dst)
                copied.append(src)

    # Remote pruning (never touches the newest keep_last).
    remote_ckpts = sorted(remote_dir.glob(CHECKPOINT_GLOB), key=_step_of)
    for stale in remote_ckpts[:-keep_last] if keep_last > 0 else []:
        with contextlib.suppress(OSError):  # Drive hiccup — best effort
            stale.unlink()
    return copied


class CheckpointSync:
    """Background mirror of a training run directory into a remote directory.

    Typical wiring (see ``finetune/run_finetune.py``)::

        sync = CheckpointSync(local_dir, remote_dir, interval_s=60)
        sync.restore_first()      # pull the newest remote checkpoint in
        sync.start()              # mirror while train.py runs
        ...
        sync.stop(final=True)     # last pass incl. ft_model.pth
    """

    def __init__(
        self,
        local_dir: Path,
        remote_dir: Path | None,
        *,
        interval_s: float = 60.0,
        keep_last: int = 3,
        log=print,
    ) -> None:
        self.local_dir = Path(local_dir)
        self.remote_dir = Path(remote_dir) if remote_dir else None
        self.interval_s = max(5.0, float(interval_s))
        self.keep_last = int(keep_last)
        self._log = log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_push: list[Path] = []

    @property
    def enabled(self) -> bool:
        return self.remote_dir is not None

    def restore_first(self) -> Path | None:
        if self.remote_dir is None:
            self._log("[sync] no remote checkpoint dir — resume is local-only")
            return latest_checkpoint(self.local_dir)
        if not self.remote_dir.is_dir():
            self._log(f"[sync] remote mirror is empty (fresh): {self.remote_dir}")
            return latest_checkpoint(self.local_dir)
        return restore(self.local_dir, self.remote_dir)

    def push(self, *, include_final: bool = False) -> list[Path]:
        remote = self.remote_dir
        if remote is None:
            return []
        remote.mkdir(parents=True, exist_ok=True)
        try:
            self.last_push = push_once(
                self.local_dir, remote,
                keep_last=self.keep_last, include_final=include_final,
            )
        except Exception as exc:  # noqa: BLE001 — never kill training for sync
            self._log(f"[sync] push failed (will retry): {exc}")
            return []
        for path in self.last_push:
            self._log(f"[sync] mirrored {path.name} → {remote}")
        return self.last_push

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="rafidain-ckpt-sync", daemon=True
        )
        self._thread.start()
        self._log(
            f"[sync] background mirror started: {self.local_dir} → "
            f"{self.remote_dir} every {self.interval_s:.0f}s "
            f"(keep_last={self.keep_last})"
        )

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.push()

    def stop(self, *, final: bool = True, timeout: float = 120.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if final:
            self.push(include_final=True)
            if self.enabled:
                self._log(f"[sync] final state mirrored to {self.remote_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _seed_root_from_env() -> Path:
    env = os.environ.get("SEED_VC_ROOT", "").strip()
    return Path(env).expanduser().resolve() if env else (
        Path(__file__).resolve().parents[1] / "third_party" / "seed-vc"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Checkpoint sync utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)

    watch = sub.add_parser("watch", help="mirror a run dir to Drive until Ctrl+C")
    watch.add_argument("--local-dir", type=Path, required=True)
    watch.add_argument("--interval", type=float, default=60.0)
    watch.add_argument("--keep-last", type=int, default=3)

    rst = sub.add_parser("restore", help="pull the newest remote checkpoint in")
    rst.add_argument("--run-name", default="iraqi_run")
    rst.add_argument("--local-dir", type=Path, default=None)

    st = sub.add_parser("status", help="show local/remote checkpoint state")
    st.add_argument("--run-name", default="iraqi_run")

    args = ap.parse_args()
    root = _seed_root_from_env()
    gdrive = detect_google_drive()

    if args.cmd == "watch":
        sync = CheckpointSync(
            args.local_dir, default_remote_root(gdrive),
            interval_s=args.interval, keep_last=args.keep_last,
        )
        sync.restore_first()
        sync.start()
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            sync.stop()
        return 0

    local = args.local_dir or run_dir_for(root, args.run_name)
    remote = default_remote_root(gdrive) / args.run_name if gdrive else None

    if args.cmd == "restore":
        ckpt = restore(local, remote)
        print(f"[restore] resume checkpoint: {ckpt or 'NONE (fresh run)'}")
        return 0

    # status
    for label, directory in (("local", local), ("remote", remote)):
        ckpt = latest_checkpoint(directory) if directory else None
        where = str(ckpt) if ckpt else f"(none in {directory})"
        print(f"[status] {label}: {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
