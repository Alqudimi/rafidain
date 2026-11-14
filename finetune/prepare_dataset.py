#!/usr/bin/env python3
"""Prepare audio recordings for Seed-VC fine-tuning.

The preparation is transactional: files are written to a fresh staging
 directory and published only after the complete pass succeeds. Existing
 outputs are never silently mixed with a new manifest; pass --overwrite to
 replace an existing output directory explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".opus", ".ogg", ".aac", ".wma"}
MODEL_SR = 22_050
FADE_MS = 10
FRAME_MS = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_any(path: Path, sample_rate: int) -> np.ndarray:
    """Decode the first audio stream to mono signed-16 PCM at sample_rate."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False, timeout=600)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg timed out on {path.name}") from exc
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace")[:300]
        raise RuntimeError(f"ffmpeg failed on {path.name}: {detail}")
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0


def peak_normalise(x: np.ndarray, target: float = 10 ** (-3 / 20)) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if not math.isfinite(peak) or peak < 1e-6:
        return x
    return (x / peak * target).astype(np.float32, copy=False)


def fade_edges(x: np.ndarray, sr: int, fade_ms: int = FADE_MS) -> np.ndarray:
    """Apply short symmetric linear fades to prevent clicks at segment edges."""
    count = min(max(0, int(round(sr * fade_ms / 1000))), x.size // 2)
    if count == 0:
        return x.astype(np.float32, copy=False)
    out = x.astype(np.float32, copy=True)
    ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
    out[:count] *= ramp
    out[-count:] *= ramp[::-1]
    return out


def trim_silence(x: np.ndarray, sr: int, threshold_db: float = -45.0) -> np.ndarray:
    """Trim only leading/trailing low-energy 20-ms frames; preserve partial frames."""
    if x.size == 0:
        return x
    win = max(1, int(round(FRAME_MS / 1000 * sr)))
    threshold = 10 ** (threshold_db / 20)
    frames = [x[i : i + win] for i in range(0, x.size, win)]
    active = [
        i
        for i, frame in enumerate(frames)
        if frame.size and float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) > threshold
    ]
    if not active:
        return x
    start = active[0] * win
    end = min(x.size, (active[-1] + 1) * win)
    return x[start:end]


def _quiet_cut(x: np.ndarray, start: int, max_len: int, sr: int) -> int:
    """Choose the quietest 20-ms boundary in the final second before max_len."""
    frame = max(1, int(round(FRAME_MS / 1000 * sr)))
    lower = max(start + frame, start + max_len - sr)
    upper = min(start + max_len, x.size)
    first = ((lower + frame - 1) // frame) * frame
    candidates = list(range(first, upper + 1, frame))
    if not candidates:
        return min(start + max_len, x.size)
    return min(
        candidates,
        key=lambda cut: (
            float(np.sqrt(np.mean(x[max(start, cut - frame) : cut].astype(np.float64) ** 2)))
            if cut > start
            else float("inf")
        ),
    )


def segment(x: np.ndarray, sr: int, max_sec: float, min_sec: float) -> list[np.ndarray]:
    """Split long audio near a quiet boundary; merge a short tail where possible."""
    max_len = int(max_sec * sr)
    min_len = int(min_sec * sr)
    if x.size < min_len:
        return []
    result: list[np.ndarray] = []
    start = 0
    while x.size - start > max_len:
        cut = _quiet_cut(x, start, max_len, sr)
        if cut - start < min_len:
            cut = start + max_len
        result.append(x[start:cut].astype(np.float32, copy=False))
        start = cut
    tail = x[start:]
    if tail.size >= min_len:
        result.append(tail.astype(np.float32, copy=False))
    elif tail.size and result:
        # Borrow just enough samples from the previous segment so neither the
        # tail nor the preceding chunk is discarded or made shorter than min.
        needed = min_len - tail.size
        previous = result[-1]
        if previous.size - needed >= min_len:
            result[-1] = previous[:-needed].astype(np.float32, copy=False)
            result.append(np.concatenate((previous[-needed:], tail)).astype(np.float32, copy=False))
        elif previous.size + tail.size <= int(max_sec * sr):
            result[-1] = np.concatenate((previous, tail)).astype(np.float32, copy=False)
    return result


def write_wav(path: Path, pcm: np.ndarray, sr: int) -> None:
    """Write mono 16-bit little-endian PCM WAV without importing optional codecs."""
    import wave

    clipped = np.clip(pcm, -1.0, 1.0 - 1 / 32768)
    raw = (clipped * 32768.0).round().astype("<i2").tobytes()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(raw)


def discover_audio(input_dir: Path, output_dir: Path) -> list[Path]:
    """Find sorted audio while excluding output and its abandoned staging siblings."""
    output_dir = output_dir.resolve()
    staging_prefix = f".{output_dir.name}.staging-"

    def is_staging_file(path: Path) -> bool:
        return any(
            parent.parent == output_dir.parent and parent.name.startswith(staging_prefix)
            for parent in path.parents
        )

    return sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTS
        and not p.is_symlink()
        and output_dir not in p.resolve().parents
        and p.resolve() != output_dir
        and not is_staging_file(p)
    )


def _publish(staging: Path, output: Path, overwrite: bool) -> None:
    backup: Path | None = None
    if not output.exists():
        prior_backups = sorted(
            (p for p in output.parent.glob(f".{output.name}.backup-*")
             if p.is_dir() and not p.is_symlink()),
            key=lambda p: p.stat().st_mtime_ns,
        )
        if prior_backups:
            # Recover the old complete dataset if the process died after moving it
            # aside but before the new staging directory was published.
            os.replace(prior_backups[-1], output)
    if output.exists():
        if overwrite and any(output.iterdir()):
            backup = Path(
                tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=output.parent)
            )
            backup.rmdir()
            os.replace(output, backup)
        else:
            output.rmdir()
    try:
        os.replace(staging, output)
    except Exception:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def prepare(
    input_dir: Path,
    output_dir: Path,
    *,
    sample_rate: int = MODEL_SR,
    min_sec: float = 1.0,
    max_sec: float = 28.0,
    overwrite: bool = False,
) -> dict[str, int | float]:
    output_path = Path(output_dir).expanduser()
    output_is_symlink = output_path.is_symlink()
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = output_path.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input directory not found: {input_dir}")
    if sample_rate < 8_000 or sample_rate > 96_000:
        raise ValueError("sample rate must be between 8000 and 96000 Hz")
    if not (0 < min_sec < max_sec <= 30):
        raise ValueError("require 0 < min-sec < max-sec <= 30")
    if output_dir == input_dir or output_dir in input_dir.parents:
        raise ValueError("output-dir cannot be the input directory or an ancestor of it")
    if output_is_symlink:
        raise ValueError("output-dir must not be a symbolic link")
    if output_dir.exists() and not overwrite and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose a new path or pass --overwrite"
        )

    files = discover_audio(input_dir, output_dir)
    if not files:
        raise ValueError(f"no supported audio files found under {input_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    records: list[dict[str, object]] = []
    skipped = 0
    total_samples = 0
    try:
        for ordinal, path in enumerate(files):
            rel = path.relative_to(input_dir).as_posix()
            try:
                before = path.stat()
                source_hash = sha256_file(path)
                pcm = peak_normalise(trim_silence(decode_any(path, sample_rate), sample_rate))
                segments = segment(pcm, sample_rate, max_sec, min_sec)
                if not segments:
                    raise ValueError("no segment remains after trimming/length validation")
                after = path.stat()
                if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                    raise RuntimeError("source changed while it was being prepared")
            except (RuntimeError, ValueError, OSError) as exc:
                print(f"[prep] skip {rel}: {exc}", file=sys.stderr)
                skipped += 1
                continue

            # Path hash prevents same-basename inputs in different folders overwriting each other.
            source_key = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:12]
            safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in path.stem)[:60]
            for index, raw_audio in enumerate(segments):
                audio = fade_edges(raw_audio, sample_rate)
                out_name = f"{safe_stem}_{source_key}_{index:04d}.wav"
                out_path = staging / out_name
                write_wav(out_path, audio, sample_rate)
                records.append(
                    {
                        "path": out_name,
                        "duration_s": round(audio.size / sample_rate, 6),
                        "source": rel,
                        "source_sha256": source_hash,
                        "segment_sha256": sha256_file(out_path),
                        "segment_index": index,
                        "sample_rate": sample_rate,
                        "channels": 1,
                        "subtype": "PCM_16",
                        "peak_target_dbfs": -3.0,
                        "trim_threshold_dbfs": -45.0,
                        "fade_ms": FADE_MS,
                    }
                )
                total_samples += audio.size
            if (ordinal + 1) % 1000 == 0 or ordinal + 1 == len(files):
                print(
                    f"[prep] processed {ordinal + 1}/{len(files)} source files; "
                    f"{len(records)} segments"
                )

        if not records:
            raise RuntimeError("all source files failed validation; no output was published")
        manifest = staging / "manifest.jsonl"
        with manifest.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        summary = {
            "schema_version": 1,
            "input_dir": str(input_dir),
            "sample_rate": sample_rate,
            "segments": len(records),
            "hours": round(total_samples / sample_rate / 3600, 6),
            "source_files": len(files),
            "skipped_sources": skipped,
            "min_sec": min_sec,
            "max_sec": max_sec,
            "fade_ms": FADE_MS,
            "trim_threshold_dbfs": -45.0,
            "peak_target_dbfs": -3.0,
            "manifest_sha256": sha256_file(manifest),
        }
        (staging / "_stats.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _publish(staging, output_dir, overwrite)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", required=True, help="raw recordings (ffmpeg-readable audio)")
    parser.add_argument("--output-dir", default=None, help="default: <input-dir>/prepared")
    parser.add_argument("--sr", type=int, default=MODEL_SR)
    parser.add_argument("--min-sec", type=float, default=1.0)
    parser.add_argument("--max-sec", type=float, default=28.0)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing output atomically"
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir / "prepared"
    )
    try:
        summary = prepare(
            input_dir,
            output_dir,
            sample_rate=args.sr,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"[prep] done: {summary['segments']} segments, {summary['hours']:.3f} h, "
        f"{summary['skipped_sources']} skipped -> {output_dir}"
    )
    print(f"[prep] manifest: {output_dir / 'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
