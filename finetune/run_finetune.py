#!/usr/bin/env python3
"""Fine-tune seed-uvit-tat-xlsr-tiny on an Iraqi-Arabic corpus.

A robust wrapper around the *upstream* ``train.py`` of Seed-VC:

1. pre-flight checks (seed-vc repo, torch, dataset);
2. optional dataset preparation (``--prepare`` -> prepare_dataset.py);
3. write a run-specific patched config (guarantees training starts from the
   published ``DiT_uvit_tat_xlsr_ema.pth`` checkpoint instead of scratch);
4. launch ``train.py`` and stream its output, emitting ``[progress] N``
   markers (parsed from step logs and saved checkpoints) for the API job.

Upstream training contract (verified against Plachtaa/seed-vc):
    dataset: audio files of 1-30 s (wav/flac/mp3/m4a/opus/ogg), speaker
    labels not required, each speaker >= 1 utterance, clean audio preferred.
    Minimum viable run: ~1 utterance/speaker, 100 steps (~2 min on a T4).

Example
    python finetune/run_finetune.py --dataset-dir ./datasets/iraqi_prepared \
        --run-name iraqi_v1 --max-steps 2000 --prepare
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESET = "configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml"
HF_CHECKPOINT = "DiT_uvit_tat_xlsr_ema.pth"
_STEP_PATTERNS = (
    re.compile(r"step[=:\s]+(\d+)"),
    re.compile(r"(\d+)\s*/\s*\d+\s*steps"),
)


def seed_vc_root() -> Path:
    root = os.environ.get("SEED_VC_ROOT", "").strip()
    return Path(root).resolve() if root else REPO_ROOT / "third_party" / "seed-vc"


def preflight(root: Path, dataset_dir: Path, need_torch: bool) -> None:
    if not (root / "train.py").exists():
        sys.exit(
            f"error: Seed-VC repo not found at {root}\n"
            "  -> run `make seedvc` or set SEED_VC_ROOT"
        )
    if not dataset_dir.is_dir():
        sys.exit(f"error: dataset dir not found: {dataset_dir}")
    audio = [
        p for p in dataset_dir.rglob("*")
        if p.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".opus", ".ogg"}
    ]
    if not audio:
        sys.exit(f"error: no audio files under {dataset_dir}")
    print(f"[preflight] {len(audio)} audio file(s) found")
    if need_torch:
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.exit("error: torch not installed — pip install -r requirements-finetune.txt")
        print("[preflight] torch OK")


def patch_config(root: Path, runs_dir: Path, pretrained_ckpt: str | None) -> Path:
    """Copy the official preset and guarantee a known starting checkpoint.

    Upstream resume order: --pretrained-ckpt > latest runs/<name>/ checkpoint
    > config['pretrained_model'] (HuggingFace) > scratch. We always populate
    `pretrained_model` so a fresh run starts from the published weights.
    """
    import yaml

    src = root / DEFAULT_PRESET
    if not src.exists():
        sys.exit(f"error: preset config not found: {src}")
    config = yaml.safe_load(src.read_text(encoding="utf-8"))
    config.setdefault("pretrained_model", HF_CHECKPOINT)
    patched = runs_dir / "config_patched.yml"
    patched.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    print(f"[config] patched preset -> {patched}")
    _ = pretrained_ckpt  # when given, upstream --pretrained-ckpt takes priority
    return patched


def emit_progress(runs_dir: Path, max_steps: int, state: dict) -> None:
    """Best-effort progress from saved checkpoints."""
    best = state.get("progress", 0.0)
    if runs_dir.exists():
        for ckpt in runs_dir.glob("DiT_epoch_*_step_*.pth"):
            match = re.search(r"step_(\d+)", ckpt.name)
            if match:
                best = max(best, min(1.0, int(match.group(1)) / max_steps))
    if best > state.get("progress", 0.0):
        state["progress"] = best
        print(f"[progress] {best:.3f}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--run-name", default="iraqi_run")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--pretrained-ckpt", default=None, help="start from this .pth")
    ap.add_argument("--prepare", action="store_true", help="run prepare_dataset.py first")
    args = ap.parse_args()

    root = seed_vc_root()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    runs_dir = REPO_ROOT / "runs" / args.run_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    preflight(root, dataset_dir, need_torch=True)

    if args.prepare:
        print("[prepare] preparing dataset (1-30 s, mono, 22.05 kHz)...")
        prepared = dataset_dir / "prepared"
        cmd = [
            sys.executable, str(REPO_ROOT / "finetune" / "prepare_dataset.py"),
            "--input-dir", str(dataset_dir),
        ]
        code = subprocess.call(cmd)
        if code != 0:
            sys.exit(f"error: dataset preparation failed ({code})")
        if (prepared / "manifest.jsonl").exists():
            print(f"[prepare] manifest: {prepared / 'manifest.jsonl'}")
            lines = (prepared / "manifest.jsonl").read_text().strip().splitlines()
            hours = sum(json.loads(line)["duration_s"] for line in lines) / 3600
            print(f"[prepare] {len(lines)} segments, {hours:.2f} h")

    config_path = patch_config(root, runs_dir, args.pretrained_ckpt)

    cmd = [
        sys.executable, str(root / "train.py"),
        "--config", str(config_path),
        "--dataset-dir", str(dataset_dir),
        "--run-name", args.run_name,
        "--batch-size", str(args.batch_size),
        "--max-steps", str(args.max_steps),
        "--max-epochs", str(args.max_epochs),
        "--save-every", str(args.save_every),
        "--num-workers", str(args.num_workers),
        "--gpu", str(args.gpu),
    ]
    if args.pretrained_ckpt:
        cmd += ["--pretrained-ckpt", args.pretrained_ckpt]

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    print("[train] " + " ".join(cmd), flush=True)

    state: dict = {"progress": 0.0}
    proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        cmd, cwd=str(root), env=env, text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        print(line, flush=True)
        for pattern in _STEP_PATTERNS:
            match = pattern.search(line)
            if match:
                state["progress"] = max(
                    state["progress"],
                    min(1.0, int(match.group(1)) / args.max_steps),
                )
        emit_progress(runs_dir, args.max_steps, state)
    code = proc.wait()

    if code != 0:
        print(f"[error] train.py exited with {code}", file=sys.stderr)
        return code

    final = runs_dir / "ft_model.pth"
    print("[progress] 1.000")
    print(f"[done] final model: {final}")
    print(
        "[hint] deploy it:\n"
        f"  RAFIDAIN__ENGINE__LOCAL_CHECKPOINT={final}\n"
        f"  RAFIDAIN__ENGINE__LOCAL_CONFIG={config_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
