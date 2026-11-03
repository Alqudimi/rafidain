#!/usr/bin/env python3
"""Pre-download every Seed-VC artefact the engine needs (bakes images, warms caches).

    python scripts/download_models.py [--cache-dir ./checkpoints] [--skip-xlsr]

Downloads (same identifiers the loader uses):
  * Plachta/Seed-VC          : DiT_uvit_tat_xlsr_ema.pth
                                config_dit_mel_seed_uvit_xlsr_tiny.yml
  * funasr/campplus          : campplus_cn_common.bin
  * FunAudioLLM/CosyVoice-300M: hift.pt
  * facebook/wav2vec2-xls-r-300m : XLSR content encoder (~1.27 GB, via transformers;
                                skipped gracefully when transformers/torch are absent)
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # live progress bars (tqdm ships with huggingface-hub)
    from tqdm import tqdm
except ImportError:  # graceful fallback: plain counters
    def tqdm(iterable=None, **kwargs):
        if iterable is not None:
            return iterable
        class _NoBar:
            def update(self, *_a):
                pass
            def set_description(self, *_a):
                pass
            def close(self):
                pass
        return _NoBar()

FILES = [
    ("Plachta/Seed-VC", "DiT_uvit_tat_xlsr_ema.pth"),
    ("Plachta/Seed-VC", "config_dit_mel_seed_uvit_xlsr_tiny.yml"),
    ("funasr/campplus", "campplus_cn_common.bin"),
    ("FunAudioLLM/CosyVoice-300M", "hift.pt"),
]

XLSR_MODEL = "facebook/wav2vec2-xls-r-300m"


def prefetch_xlsr() -> bool:
    """Warm the transformers cache with the XLSR content encoder.

    The loader calls ``Wav2Vec2Model.from_pretrained(XLSR_MODEL)`` at engine
    load time; pre-fetching here turns the first server start from a
    ~1.3 GB cold download into a warm cache hit.
    """
    try:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    except Exception as exc:  # ImportError (transformers/torch missing) or backend errors
        print(f"SKIP {XLSR_MODEL}: transformers stack unavailable ({exc.__class__.__name__}).")
        print("     It will be downloaded automatically on first server start instead.")
        return False
    print(f"... {XLSR_MODEL} (~1.27 GB, one-time download) — this may take a few minutes")
    Wav2Vec2FeatureExtractor.from_pretrained(XLSR_MODEL)
    Wav2Vec2Model.from_pretrained(XLSR_MODEL)
    print(f"OK  {XLSR_MODEL} (XLSR content encoder) cached")
    return True


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", default="./checkpoints")
    ap.add_argument(
        "--skip-xlsr",
        action="store_true",
        help="skip the XLSR content encoder (it will download on first start)",
    )
    args = ap.parse_args()
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    files_bar = tqdm(FILES, desc="checkpoints", unit="file", leave=True)
    for repo_id, filename in files_bar:
        files_bar.set_description(f"checkpoints: {repo_id}/{filename}")
        path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache))
        size_mb = Path(path).stat().st_size / 1e6
        print(f"OK  {repo_id}/{filename}  ({_human(size_mb * 1e6)}) -> {path}")
    files_bar.close()
    if not args.skip_xlsr:
        prefetch_xlsr()
    print("\nAll checkpoints ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
