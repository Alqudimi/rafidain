"""Robust audio decoding via ffmpeg subprocess.

Accepts any container/codec ffmpeg understands (wav, mp3, flac, m4a, opus,
ogg...) and returns mono float32 PCM at the requested rate. Using ffmpeg as
an external decoder keeps the core dependency tree light and the attack
surface small (no parsing of untrusted containers in Python).
"""

from __future__ import annotations

import subprocess

import numpy as np

from rafidain.core.errors import VoiceValidationError


def decode_to_pcm(
    data: bytes,
    *,
    sample_rate: int = 16_000,
    ffmpeg_bin: str = "ffmpeg",
) -> np.ndarray:
    """Decode arbitrary audio bytes to mono float32 PCM at ``sample_rate``."""
    if not data:
        raise VoiceValidationError("empty audio payload")
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-vn",  # ignore video streams
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd, input=data, capture_output=True, timeout=30, check=False
        )
    except FileNotFoundError as exc:
        raise VoiceValidationError("ffmpeg binary not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise VoiceValidationError("audio decoding timed out") from exc
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace")[:400]
        raise VoiceValidationError(f"could not decode audio: {detail}")
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0


def encode_wav_pcm16(pcm: np.ndarray, sample_rate: int, path) -> None:
    """Write mono float PCM as a 16-bit PCM WAV file (stdlib ``wave``)."""
    import wave

    clipped = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0 - 1e-6)
    raw = (clipped * 32768.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(raw)


def read_wav_pcm16(path) -> np.ndarray:
    """Read a mono 16-bit PCM WAV file as float32."""
    import wave

    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise VoiceValidationError(f"{path}: expected mono 16-bit PCM WAV")
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    del rate
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
