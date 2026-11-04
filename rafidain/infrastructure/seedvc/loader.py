"""Loads the Seed-VC model stack for seed-uvit-tat-xlsr-tiny.

This module is the *only* place that knows the upstream repository layout of
https://github.com/Plachtaa/seed-vc. It faithfully mirrors the official
loading path used by ``real-time-gui.py`` / ``inference.py``:

1. DiT (U-ViT) flow-matching model  — ``Plachta/Seed-VC``
   ``DiT_uvit_tat_xlsr_ema.pth`` + ``config_dit_mel_seed_uvit_xlsr_tiny.yml``
2. Content encoder — XLSR (``facebook/wav2vec2-xls-r-300m``, truncated to
   the layer count declared in the model config).
3. Speaker encoder — CAMPPlus (``funasr/campplus`` / ``campplus_cn_common.bin``).
4. Vocoder — HiFTGenerator (``FunAudioLLM/CosyVoice-300M`` / ``hift.pt`` with
   ``configs/hifigan.yml`` from the seed-vc repository).

All heavyweight imports (torch, transformers, seed-vc internals) happen
lazily inside functions so that the rest of the service — and the whole test
suite — runs without the ML stack installed.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rafidain.core.config import EngineSettings, seed_vc_root
from rafidain.core.errors import ConfigError, DependencyMissingError, EngineError
from rafidain.core.logging import get_logger

logger = get_logger(__name__)

#: Fixed content-encoder vs DiT frame offset used by upstream real-time code.
CE_DIT_DIFFERENCE_S = 2.0


@dataclass(slots=True)
class SeedVCBundle:
    """Everything needed to run streaming conversions, loaded once per process."""

    model: Any
    semantic_fn: Callable[..., Any]
    vocoder_fn: Any
    campplus: Any
    to_mel: Callable[..., Any]
    kaldi: Any  # torchaudio.compliance.kaldi
    resample_up: Any  # 16k -> model rate
    mel_fn_args: dict
    sr: int
    hop_length: int
    device: Any
    fp16: bool
    version_info: dict


def _import_seed_vc(root: Path) -> Any:
    """Import the seed-vc repo (not packaged on PyPI) by adding it to sys.path."""
    if not (root / "modules" / "commons.py").exists():
        raise DependencyMissingError(
            "Seed-VC repository not found. Clone it and point SEED_VC_ROOT at it:\n"
            "  make seedvc          # clones to third_party/seed-vc\n"
            "  make install ENGINE=seedvc",
            detail={"expected_root": str(root)},
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    import types

    from modules.audio import mel_spectrogram
    from modules.campplus.DTDNN import CAMPPlus
    from modules.commons import build_model, load_checkpoint, recursive_munch
    from modules.hifigan.f0_predictor import ConvRNNF0Predictor
    from modules.hifigan.generator import HiFTGenerator

    return types.SimpleNamespace(
        mel_spectrogram=mel_spectrogram,
        CAMPPlus=CAMPPlus,
        build_model=build_model,
        load_checkpoint=load_checkpoint,
        recursive_munch=recursive_munch,
        ConvRNNF0Predictor=ConvRNNF0Predictor,
        HiFTGenerator=HiFTGenerator,
    )


def _hf_download(repo_id: str, filename: str, cache_dir: Path) -> Path:
    """Download (or reuse the cached copy of) a single HuggingFace file."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - env specific
        raise DependencyMissingError(
            "huggingface-hub is required to fetch Seed-VC checkpoints: "
            "make install ENGINE=seedvc"
        ) from exc
    path = Path(
        hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache_dir))
    )
    logger.info("checkpoint ready: %s/%s -> %s", repo_id, filename, path)
    return path


def resolve_device(spec: str) -> Any:
    """Resolve the torch device from a setting string."""
    import torch

    if spec == "cuda":
        if not torch.cuda.is_available():
            raise ConfigError("device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "mps":
        return torch.device("mps")
    if spec != "auto":
        raise ConfigError(f"unknown device spec: {spec!r}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    logger.warning("no GPU found — falling back to CPU (expect high latency)")
    return torch.device("cpu")


def load_bundle(cfg: EngineSettings) -> SeedVCBundle:
    """Load the complete seed-uvit-tat-xlsr-tiny stack. Heavy: call once."""
    import torch
    import torchaudio

    root = seed_vc_root()
    sv = _import_seed_vc(root)
    device = resolve_device(cfg.device)
    fp16 = cfg.fp16 and device.type == "cuda"

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. DiT / flow-matching model -------------------------------------
    if cfg.local_checkpoint:
        ckpt_path = Path(cfg.local_checkpoint)
        if not ckpt_path.exists():
            raise ConfigError(f"local checkpoint not found: {ckpt_path}")
        config_path = (
            Path(cfg.local_config)
            if cfg.local_config
            else root / "configs" / "presets" / "config_dit_mel_seed_uvit_xlsr_tiny.yml"
        )
    else:
        ckpt_path = _hf_download(cfg.hf_repo, cfg.checkpoint, cfg.cache_dir)
        config_path = _hf_download(cfg.hf_repo, cfg.config, cfg.cache_dir)

    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    model_params = sv.recursive_munch(config["model_params"])
    model = sv.build_model(model_params, stage="DiT")
    model, _, _, _ = sv.load_checkpoint(
        model, None, str(ckpt_path),
        load_only_params=True, ignore_modules=[], is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    # Streaming KV caches for the U-ViT estimator (upstream requirement).
    try:
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    except AttributeError:
        logger.warning("cfm.estimator.setup_caches not found — upstream API drift?")

    spect = config["preprocess_params"]["spect_params"]
    sr = int(config["preprocess_params"]["sr"])
    hop_length = int(spect["hop_length"])

    # --- 2. Content encoder (XLSR) -----------------------------------------
    tokenizer_cfg = config["model_params"]["speech_tokenizer"]
    if tokenizer_cfg.get("type") != "xlsr":
        raise ConfigError(
            "this adapter targets seed-uvit-tat-xlsr-tiny (XLSR tokenizer); "
            f"config declares {tokenizer_cfg.get('type')!r}"
        )
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    xlsr_name = tokenizer_cfg["name"]
    output_layer = tokenizer_cfg.get("output_layer")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(xlsr_name)
    wav2vec = Wav2Vec2Model.from_pretrained(xlsr_name)
    if output_layer:
        wav2vec.encoder.layers = wav2vec.encoder.layers[: int(output_layer)]
    wav2vec = wav2vec.to(device).eval()
    if fp16:
        wav2vec = wav2vec.half()

    def semantic_fn(waves_16k):
        """(1, T) float tensor at 16 kHz -> content features (1, T', D)."""
        inputs = feature_extractor(
            [waves_16k.squeeze(0).cpu().numpy()],
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            sampling_rate=16_000,
        )
        values = inputs.input_values.to(device)
        if fp16:
            values = values.half()
        with torch.no_grad():
            outputs = wav2vec(values)
        return outputs.last_hidden_state.float()

    # --- 3. Speaker encoder (CAMPPlus) --------------------------------------
    campplus = sv.CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_path = _hf_download(cfg.campplus_repo, cfg.campplus_file, cfg.cache_dir)
    campplus.load_state_dict(torch.load(str(campplus_path), map_location="cpu"))
    campplus.eval().to(device)

    # --- 4. Vocoder (HiFT) ----------------------------------------------------
    vocoder_type = model_params.vocoder.type
    if vocoder_type != "hifigan":
        raise ConfigError(
            f"seed-uvit-tat-xlsr-tiny expects the HiFT vocoder, got {vocoder_type!r}"
        )
    hift_yaml = root / "configs" / "hifigan.yml"
    with open(hift_yaml, encoding="utf-8") as fh:
        hift_cfg = yaml.safe_load(fh)
    hift = sv.HiFTGenerator(
        **hift_cfg["hift"],
        f0_predictor=sv.ConvRNNF0Predictor(**hift_cfg["f0_predictor"]),
    )
    hift_path = _hf_download(cfg.hift_repo, cfg.hift_file, cfg.cache_dir)
    hift.load_state_dict(torch.load(str(hift_path), map_location="cpu"))
    hift.eval().to(device)

    # --- 5. Mel front-end ------------------------------------------------------
    mel_fn_args = {
        "n_fft": int(spect["n_fft"]),
        "win_size": int(spect["win_length"]),
        "hop_size": int(spect["hop_length"]),
        "num_mels": int(spect["n_mels"]),
        "sampling_rate": sr,
        "fmin": spect.get("fmin", 0),
        "fmax": None if spect.get("fmax", "None") == "None" else 8000,
        "center": False,
    }

    def to_mel(x):
        return sv.mel_spectrogram(x, **mel_fn_args)

    bundle = SeedVCBundle(
        model=model,
        semantic_fn=semantic_fn,
        vocoder_fn=hift,
        campplus=campplus,
        to_mel=to_mel,
        kaldi=torchaudio.compliance.kaldi,
        resample_up=torchaudio.transforms.Resample(16_000, sr).to(device),
        mel_fn_args=mel_fn_args,
        sr=sr,
        hop_length=hop_length,
        device=device,
        fp16=fp16,
        version_info={
            "checkpoint": str(ckpt_path),
            "config": str(config_path),
            "device": str(device),
            "fp16": fp16,
            "sr": sr,
            "xlsr": xlsr_name,
        },
    )
    logger.info("Seed-VC bundle loaded: %s", bundle.version_info)
    return bundle


__all__ = ["CE_DIT_DIFFERENCE_S", "SeedVCBundle", "load_bundle", "resolve_device"]

# Re-exported lazily used by engine; keep import error surface here.
_ = EngineError
