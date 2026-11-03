"""Offline quality evaluation of converted speech (needs the ML stack)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2, "mono 16-bit WAV"
        sr = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, sr


def load_speaker_encoder():
    """CAMPPlus via the seed-vc repo (same model the engine conditions on)."""
    import os
    import sys as _sys

    root = Path(os.environ.get("SEED_VC_ROOT", REPO_ROOT / "third_party" / "seed-vc"))
    if str(root) not in _sys.path:
        _sys.path.insert(0, str(root))
    import torch
    import torchaudio
    from huggingface_hub import hf_hub_download
    from modules.campplus.DTDNN import CAMPPlus

    model = CAMPPlus(feat_dim=80, embedding_size=192)
    ckpt = hf_hub_download(
        repo_id="funasr/campplus", filename="campplus_cn_common.bin",
        cache_dir=str(REPO_ROOT / "checkpoints"),
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval(), torchaudio.compliance.kaldi


def embed(model, kaldi, pcm: np.ndarray, sr: int) -> np.ndarray:
    import torch

    if sr != 16_000:
        import soxr

        pcm = soxr.resample(pcm, sr, 16_000)
    pcm_t = torch.from_numpy(pcm.astype(np.float32)).reshape(1, -1)
    feat = kaldi.fbank(pcm_t, num_mel_bins=80, dither=0, sample_frequency=16_000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    with torch.no_grad():
        return model(feat.unsqueeze(0)).squeeze(0).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a converted sample")
    ap.add_argument("--source", required=True, help="original (input) speech WAV")
    ap.add_argument("--reference", required=True, help="target-voice reference WAV")
    ap.add_argument("--converted", required=True, help="model output WAV")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    src, sr_s = read_wav(Path(args.source))
    ref, sr_r = read_wav(Path(args.reference))
    out, sr_o = read_wav(Path(args.converted))

    report: dict = {
        "source": args.source,
        "reference": args.reference,
        "converted": args.converted,
        "durations_s": {
            "source": round(src.size / sr_s, 3),
            "converted": round(out.size / sr_o, 3),
            "ratio": round((out.size / sr_o) / (src.size / sr_s), 4),
        },
        "levels_dbfs": {
            "source": round(20 * np.log10(np.sqrt(np.mean(src**2)) + 1e-12), 2),
            "converted": round(20 * np.log10(np.sqrt(np.mean(out**2)) + 1e-12), 2),
        },
        "peak_clipping": {
            "clipped_samples": int(np.sum(np.abs(out) >= 0.999)),
        },
    }

    # Speaker similarity (SECS-style): cosine(converted, target reference)
    try:
        model, kaldi = load_speaker_encoder()
        e_out = embed(model, kaldi, out, sr_o)
        e_ref = embed(model, kaldi, ref, sr_r)
        e_src = embed(model, kaldi, src, sr_s)
        def cos(a, b) -> float:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        report["speaker_similarity"] = {
            "converted_vs_reference": round(cos(e_out, e_ref), 4),
            "converted_vs_source": round(cos(e_out, e_src), 4),
            "source_vs_reference": round(cos(e_src, e_ref), 4),
        }
        # Interpretation: converted_vs_reference should approach
        # source_vs_reference's complement: the output should sound like the
        # target, not the source.
    except Exception as exc:  # noqa: BLE001 — keep the report useful without torch
        report["speaker_similarity"] = {"unavailable": f"{type(exc).__name__}: {exc}"}

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
