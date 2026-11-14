#!/usr/bin/env python3
"""Run a guarded Seed-VC fine-tuning job.

The wrapper keeps raw inputs separate from the exact directory sent to the
upstream loader, refuses accidental run/checkpoint reuse, and supports explicit
full-state resume when using compatible Seed-VC rolling checkpoints.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_PRESET = "configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml"
HF_CHECKPOINT = "DiT_uvit_tat_xlsr_ema.pth"
_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".opus", ".ogg"}
_PREP_AUDIO_EXTS = _AUDIO_EXTS | {".aac", ".wma"}
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_STEP_PATTERNS = (
    re.compile(r"step[=:\s]+(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)\s*/\s*\d+\s*steps", re.IGNORECASE),
)


def seed_vc_root() -> Path:
    root = os.environ.get("SEED_VC_ROOT", "").strip()
    return Path(root).expanduser().resolve() if root else REPO_ROOT / "third_party" / "seed-vc"


def audio_files(directory: Path, *, for_prepare: bool = False) -> list[Path]:
    extensions = _PREP_AUDIO_EXTS if for_prepare else _AUDIO_EXTS
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and (p.suffix.lower() in extensions if for_prepare else p.suffix in extensions)
    )


def run_dir_has_artifacts(directory: Path) -> bool:
    """Treat any non-empty or symlinked run directory as an existing run."""
    return directory.is_symlink() or (
        directory.exists() and (not directory.is_dir() or any(directory.iterdir()))
    )


def preflight(root: Path, dataset_dir: Path, need_torch: bool, *, for_prepare: bool = False) -> int:
    if not (root / "train.py").is_file():
        raise FileNotFoundError(
            f"Seed-VC repo not found at {root}; run `make seedvc` or set SEED_VC_ROOT"
        )
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"dataset dir not found: {dataset_dir}")
    audio = audio_files(dataset_dir, for_prepare=for_prepare)
    if not audio:
        raise ValueError(f"no supported audio files under {dataset_dir}")
    if need_torch:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "torch not installed; install the Seed-VC training dependencies"
            ) from exc
    print(f"[preflight] {len(audio)} audio file(s) found under {dataset_dir}")
    return len(audio)


def patch_config(root: Path, runs_dir: Path) -> Path:
    """Copy the pinned upstream preset, retaining its architecture and defaults."""
    import yaml

    src = root / DEFAULT_PRESET
    if not src.is_file():
        raise FileNotFoundError(f"preset config not found: {src}")
    config = yaml.safe_load(src.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"invalid YAML preset (expected mapping): {src}")
    config.setdefault("pretrained_model", HF_CHECKPOINT)
    patched = runs_dir / "config_patched.yml"
    temp = patched.with_suffix(".yml.tmp")
    temp.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    temp.replace(patched)
    print(f"[config] patched preset -> {patched}")
    return patched


def emit_progress(runs_dir: Path, max_steps: int, state: dict[str, float]) -> None:
    best = state.get("progress", 0.0)
    for ckpt in runs_dir.glob("DiT_epoch_*_step_*.pth"):
        match = re.search(r"step_(\d+)", ckpt.name)
        if match:
            best = max(best, min(1.0, int(match.group(1)) / max_steps))
    if best > state.get("progress", 0.0):
        state["progress"] = best
        print(f"[progress] {best:.3f}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="raw input directory with --prepare, otherwise training data",
    )
    parser.add_argument("--run-name", default="iraqi_run")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--pretrained-ckpt", default=None, help="explicit weights-only initialization checkpoint"
    )
    parser.add_argument(
        "--prepare", action="store_true", help="prepare raw audio into <dataset-dir>/prepared first"
    )
    parser.add_argument(
        "--overwrite-prepared",
        action="store_true",
        help="atomically replace an existing prepared/ folder",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume full optimizer/scheduler/step state from a run checkpoint",
    )
    parser.add_argument("--gdrive", choices=("auto", "none"), default="none")
    parser.add_argument("--gdrive-dir", default=None)
    parser.add_argument("--sync-interval", type=float, default=60.0)
    parser.add_argument("--keep-last", type=int, default=3)
    args = parser.parse_args()
    if not _RUN_NAME_RE.fullmatch(args.run_name) or args.run_name in {".", ".."}:
        parser.error(
            "run-name must be 1-64 safe ASCII letters/digits/dot/underscore/hyphen, "
            "starting alphanumeric"
        )
    if args.max_steps < 1 or args.max_epochs < 1 or args.save_every < 1:
        parser.error("max-steps, max-epochs, and save-every must be positive")
    if args.batch_size < 1 or args.num_workers < 0 or args.gpu < 0:
        parser.error("batch-size must be positive; num-workers and gpu must be non-negative")
    if args.sync_interval <= 0 or args.keep_last < 1:
        parser.error("sync-interval and keep-last must be positive")
    if args.resume and args.pretrained_ckpt:
        parser.error("--resume and --pretrained-ckpt are mutually exclusive")
    if args.resume and args.prepare:
        parser.error("--resume requires an already-prepared dataset; do not combine with --prepare")
    return args


def _run_child(
    cmd: list[str],
    *,
    root: Path,
    env: dict[str, str],
    log_path: Path,
    runs_dir: Path,
    max_steps: int,
    sync,
) -> int:
    print("[train] " + " ".join(cmd), flush=True)
    state: dict[str, float] = {"progress": 0.0}
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        text=True,
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    try:
        with log_path.open("a", encoding="utf-8") as log:
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
                for pattern in _STEP_PATTERNS:
                    match = pattern.search(line)
                    if match:
                        state["progress"] = max(
                            state.get("progress", 0.0), min(1.0, int(match.group(1)) / max_steps)
                        )
                emit_progress(runs_dir, max_steps, state)
        return proc.wait()
    except KeyboardInterrupt:
        print(
            "[train] interruption requested; forwarding SIGINT to Seed-VC",
            file=sys.stderr,
            flush=True,
        )
        proc.send_signal(signal.SIGINT)
        try:
            return proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                return proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                return proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if sync:
            sync.stop(final=True)


def main() -> int:
    args = _parse_args()
    root = seed_vc_root()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    runs_dir = root / "runs" / args.run_name

    try:
        preflight(root, dataset_dir, need_torch=True, for_prepare=args.prepare)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    train_dataset_dir = dataset_dir
    if args.prepare:
        from finetune.prepare_dataset import prepare

        train_dataset_dir = dataset_dir / "prepared"
        try:
            summary = prepare(dataset_dir, train_dataset_dir, overwrite=args.overwrite_prepared)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"error: dataset preparation failed: {exc}", file=sys.stderr)
            return 2
        print(
            f"[prepare] {summary['segments']} segments, {summary['hours']:.3f} h; "
            f"manifest: {train_dataset_dir / 'manifest.jsonl'}"
        )
        if summary["skipped_sources"]:
            print(
                f"[prepare] WARNING: {summary['skipped_sources']} of "
                f"{summary['source_files']} sources were skipped; review stderr above."
            )
    elif (dataset_dir / "prepared").is_dir():
        prepared_root = (dataset_dir / "prepared").resolve()
        direct_audio = [
            p for p in audio_files(dataset_dir) if prepared_root not in p.resolve().parents
        ]
        nested_audio = audio_files(dataset_dir / "prepared")
        if direct_audio and nested_audio:
            print(
                "error: dataset-dir contains both raw audio and prepared/ audio. "
                "Pass the prepared/ directory directly or use --prepare.",
                file=sys.stderr,
            )
            return 2
        if not direct_audio and nested_audio:
            train_dataset_dir = dataset_dir / "prepared"
            print(f"[data] using prepared subdirectory: {train_dataset_dir}")

    try:
        preflight(root, train_dataset_dir, need_torch=False)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: training data preflight failed: {exc}", file=sys.stderr)
        return 2

    # Import here to keep CLI help usable without torch-dependent runtime imports.
    from finetune.checkpoint_sync import (
        CheckpointSync,
        default_remote_root,
        detect_google_drive,
        latest_checkpoint,
    )

    had_local_run_artifacts = run_dir_has_artifacts(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    sync = None
    if args.gdrive != "none" or args.gdrive_dir:
        gdrive = (
            Path(args.gdrive_dir).expanduser().resolve()
            if args.gdrive_dir
            else detect_google_drive()
        )
        remote = (default_remote_root(gdrive) / args.run_name) if gdrive else None
        sync = CheckpointSync(
            runs_dir,
            remote,
            interval_s=args.sync_interval,
            keep_last=args.keep_last,
        )

    local_ckpt = latest_checkpoint(runs_dir)
    remote_ckpt = latest_checkpoint(sync.remote_dir) if sync and sync.remote_dir else None
    had_remote_run_artifacts = bool(
        sync and sync.remote_dir and run_dir_has_artifacts(sync.remote_dir)
    )
    if args.resume:
        if local_ckpt is None and remote_ckpt is None:
            print(
                "error: --resume requested but no rolling checkpoint exists locally or remotely",
                file=sys.stderr,
            )
            return 2
        try:
            from finetune.upstream_patch import ensure_full_state_resume

            patched = ensure_full_state_resume(root)
        except (OSError, RuntimeError) as exc:
            print(f"error: cannot enable verified full-state resume: {exc}", file=sys.stderr)
            return 2
        print(
            "[resume] "
            + (
                "enabled full-state resume in upstream train.py"
                if patched
                else "full-state resume patch already present"
            )
        )
        if sync:
            sync.restore_first()
    elif had_local_run_artifacts or had_remote_run_artifacts:
        print(
            "error: this run-name already has run artifacts. Use --resume only if a rolling "
            "training checkpoint exists, or choose a new --run-name for a fresh run.",
            file=sys.stderr,
        )
        return 2

    try:
        config_path = patch_config(root, runs_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: config preparation failed: {exc}", file=sys.stderr)
        return 2

    if args.pretrained_ckpt and not Path(args.pretrained_ckpt).expanduser().is_file():
        print(f"error: pretrained checkpoint not found: {args.pretrained_ckpt}", file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        str(root / "train.py"),
        "--config",
        str(config_path),
        "--dataset-dir",
        str(train_dataset_dir),
        "--run-name",
        args.run_name,
        "--batch-size",
        str(args.batch_size),
        "--max-steps",
        str(args.max_steps),
        "--max-epochs",
        str(args.max_epochs),
        "--save-every",
        str(args.save_every),
        "--num-workers",
        str(args.num_workers),
        "--gpu",
        str(args.gpu),
    ]
    if args.pretrained_ckpt:
        cmd += ["--pretrained-ckpt", str(Path(args.pretrained_ckpt).expanduser().resolve())]

    if sync:
        sync.start()
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    code = _run_child(
        cmd,
        root=root,
        env=env,
        log_path=runs_dir / "train_console.log",
        runs_dir=runs_dir,
        max_steps=args.max_steps,
        sync=sync,
    )
    if code != 0:
        print(f"[error] train.py exited with {code}", file=sys.stderr)
        return code

    final = runs_dir / "ft_model.pth"
    if not final.is_file():
        print(
            f"error: upstream exited successfully but final checkpoint is missing: {final}",
            file=sys.stderr,
        )
        return 3
    print("[progress] 1.000")
    print(f"[done] final model: {final}")
    print(f"[hint] deploy with RAFIDAIN__ENGINE__LOCAL_CHECKPOINT={final}")
    print(f"[hint] config: RAFIDAIN__ENGINE__LOCAL_CONFIG={config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
