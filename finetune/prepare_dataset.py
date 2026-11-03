#!/usr/bin/env python3
"""Prepare an Iraqi-Arabic speech corpus for Seed-VC fine-tuning.

Turns a folder of arbitrary recordings into the format Seed-VC's ``train.py``
expects: clean WAV files, mono, model sample rate, each 1–30 s.

Pipeline per file
    ffmpeg decode -> mono float32 @ 22.05 kHz
    -> peak-normalise to -3 dBFS
    -> trim leading/trailing silence (energy gate)
    -> split into <= --max-sec segments (50 ms fade-in/out each)
    -> skip segments shorter than --min-sec
    -> write 16-bit PCM WAV + manifest.jsonl

Usage
    python finetune/prepare_dataset.py --input-dir ./raw_iraqi \
        --output-dir ./datasets/iraqi_prepared
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".opus", ".ogg", ".aac", ".wma"}
MODEL_SR = 22_050
FADE_MS = 50


def decode_any(path: Path, sample_rate: int) -> np.ndarray:
    """Decode via ffmpeg subprocess (same strategy as the API's decode layer)."""
    import subprocess

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(
            f"ffmpeg failed on {path.name}: {proc.stderr.decode('utf-8', 'replace')[:200]}"
        )
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0


def peak_normalise(x: np.ndarray, target: float = 0.7) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 1e-6:
        return x
    return (x / peak) * target


def trim_silence(x: np.ndarray, sr: int, threshold_db: float = -45.0) -> np.ndarray:
    if x.size == 0:
        return x
    win = int(0.02 * sr)
    thresh = 10 ** (threshold_db / 20)
    n = x.size // win
    if n == 0:
        return x
    rms = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1))
    speech = np.nonzero(rms > thresh)[0]
    if speech.size == 0:
        return x
    start = int(speech[0]) * win
    end = min(x.size, (int(speech[-1]) + 1) * win)
    return x[start:end]


def segment(x: np.ndarray, sr: int, max_sec: float, min_sec: float) -> list[np.ndarray]:
    fade = int(FADE_MS / 1000 * sr)
    max_len = int(max_sec * sr)
    min_len = int(min_sec * sr)
    segments: list[np.ndarray] = []
    for start in range(0, max(x.size, 1), max_len):
        chunk = x[start : start + max_len]
        if chunk.size < min_len:
            break
        if fade > 0 and chunk.size > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            chunk = chunk.copy()
            chunk[:fade] *= ramp
            chunk[-fade:] *= ramp[::-1]
        segments.append(chunk.astype(np.float32))
    return segments


def write_wav(path: Path, pcm: np.ndarray, sr: int) -> None:
    clipped = np.clip(pcm, -1.0, 1.0 - 1e-6)
    raw = (clipped * 32768.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-dir", required=True, help="raw recordings (any ffmpeg-readable audio)")
    ap.add_argument("--output-dir", default=None, help="default: <input-dir>/prepared")
    ap.add_argument("--sr", type=int, default=MODEL_SR)
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=28.0)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = input_dir / "prepared"
    if not input_dir.is_dir():
        print(f"error: input dir not found: {input_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        and output_dir not in p.parents
    )
    print(f"[prep] found {len(files)} audio file(s) under {input_dir}")

    manifest_path = output_dir / "manifest.jsonl"
    total_sec = 0.0
    kept = 0
    skipped = 0
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for path in files:
            try:
                pcm = decode_any(path, args.sr)
            except RuntimeError as exc:
                print(f"[prep] skip {path.name}: {exc}", file=sys.stderr)
                skipped += 1
                continue
            pcm = peak_normalise(trim_silence(pcm, args.sr))
            if pcm.size < int(args.min_sec * args.sr):
                print(f"[prep] skip {path.name}: too short after trim", file=sys.stderr)
                skipped += 1
                continue
            for idx, seg in enumerate(segment(pcm, args.sr, args.max_sec, args.min_sec)):
                out_name = f"{path.stem}_{idx:03d}.wav"
                out_path = output_dir / out_name
                write_wav(out_path, seg, args.sr)
                duration = seg.size / args.sr
                total_sec += duration
                kept += 1
                manifest.write(
                    json.dumps(
                        {
                            "path": out_name,
                            "duration_s": round(duration, 3),
                            "source": path.name,
                            "sha1": hashlib.sha1(
                                (seg * 32768).astype("<i2").tobytes()
                            ).hexdigest()[:12],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(
        f"[prep] done: {kept} segment(s), {total_sec / 3600:.2f} h, "
        f"{skipped} skipped -> {output_dir}"
    )
    print("[prep] manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
