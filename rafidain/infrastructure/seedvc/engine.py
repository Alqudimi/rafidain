"""Streaming voice-conversion engine for seed-uvit-tat-xlsr-tiny.

The per-block mathematics is ported from the upstream ``real-time-gui.py``
(``custom_infer`` + SOLA crossfade), restructured into a testable, per-session
object with explicit buffer ownership:

Input side (per block of ``block_ms`` at 16 kHz):
    shift a sliding window -> XLSR content features -> length-regulated
    conditioning -> flow-matching (CFM) conditioned on (prompt, mel2, style2)
    -> HiFT vocoder -> take the block's slice (skip_head/skip_tail) ->
    equal-power crossfade with the previous chunk's tail.

Latency model (matches the upstream-published measurement):
    block_ms + crossfade_ms + context_right_ms + inference time (+ network)
    e.g. balanced profile: ~180 + 40 + 20 + 150 ≈ 390-430 ms on GPU.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from rafidain.core.config import EngineSettings
from rafidain.core.errors import ConfigError, EngineError, EngineNotReadyError, ReferenceAudioError
from rafidain.core.logging import get_logger
from rafidain.domain.value_objects import (
    MODEL_FRAMES_PER_SECOND,
    SAMPLE_RATE_IN,
    LatencyProfile,
)
from rafidain.infrastructure.seedvc.loader import (
    CE_DIT_DIFFERENCE_S,
    SeedVCBundle,
    load_bundle,
)

logger = get_logger(__name__)

_MIN_CONTEXT_LEFT_MS = 2_000  # XLSR needs > 2.0 s before frame slicing
_MIN_REF_SECONDS = 0.5


class SeedVCConverterSession:
    """One live conversion stream, conditioned on a single target voice."""

    def __init__(
        self,
        bundle: SeedVCBundle,
        ref_pcm_16k: np.ndarray,
        ref_pcm_model_sr: np.ndarray,
        profile: LatencyProfile,
        inference_slots: threading.Semaphore,
    ) -> None:
        import torch

        self._torch = torch
        self._b = bundle
        self._slots = inference_slots
        self._profile = profile
        self._closed = False

        sr = bundle.sr
        zc = sr // MODEL_FRAMES_PER_SECOND  # samples per 20 ms frame (441 @ 22050)

        def q(ms: int) -> int:
            """Quantize a duration in ms to whole model frames."""
            return int(round(ms / 1000.0 * sr / zc)) * zc

        if profile.context_left_ms < _MIN_CONTEXT_LEFT_MS:
            raise ConfigError(
                f"profile {profile.key}: context_left_ms must be >= "
                f"{_MIN_CONTEXT_LEFT_MS} for the XLSR content encoder"
            )

        self.block_frame = q(profile.block_ms)
        self.block_frame_16k = 320 * self.block_frame // zc
        self.crossfade_frame = q(profile.crossfade_ms)
        self.extra_frame = q(profile.context_left_ms)
        self.extra_frame_right = q(profile.context_right_ms)
        self.skip_head = self.extra_frame // zc
        self.skip_tail = self.extra_frame_right // zc
        self.return_length = (
            self.block_frame + self.crossfade_frame + zc
        ) // zc  # in 20 ms units
        self.ce_dit_frames = int(CE_DIT_DIFFERENCE_S * MODEL_FRAMES_PER_SECOND)

        window = (
            self.extra_frame + self.crossfade_frame + zc + self.block_frame
            + self.extra_frame_right
        )
        self.input_wav = torch.zeros(window, device=bundle.device, dtype=torch.float32)
        self.input_wav_res = torch.zeros(
            320 * window // zc, device=bundle.device, dtype=torch.float32
        )

        # Output crossfade bookkeeping: consecutive converted chunks overlap by
        # (L - block) samples; we emit exactly `block` net-new samples per call.
        self.out_len = self.return_length * sr // MODEL_FRAMES_PER_SECOND
        self.overlap = self.out_len - self.block_frame
        if self.overlap <= 0 or self.block_frame <= self.overlap:
            raise ConfigError(
                f"profile {profile.key}: block_ms too small for crossfade "
                f"(block={self.block_frame}, overlap={self.overlap} samples)"
            )
        ramp = torch.sin(
            0.5
            * float(np.pi)
            * torch.linspace(0.0, 1.0, steps=self.overlap, device=bundle.device)
        ) ** 2
        self.fade_in = ramp.float()
        self.fade_out = (1 - ramp).float()
        self._carry: Any = None
        fin = self.fade_in.detach().cpu().numpy().astype(np.float32)
        self._fade_in_np = fin
        self._fade_out_np = 1.0 - fin

        self._pending = np.zeros(0, dtype=np.float32)

        # --- condition on the reference voice (upstream prompt computation) ---
        ref16 = np.asarray(ref_pcm_16k, dtype=np.float32).reshape(-1)
        if ref16.size < int(SAMPLE_RATE_IN * _MIN_REF_SECONDS):
            raise ReferenceAudioError(
                f"reference audio too short: {ref16.size / SAMPLE_RATE_IN:.2f}s "
                f"< {_MIN_REF_SECONDS}s"
            )
        max_ref = int(SAMPLE_RATE_IN * profile.max_prompt_s)
        ref16 = ref16[:max_ref]
        ref16_t = torch.from_numpy(ref16).to(bundle.device)
        if ref_pcm_model_sr is not None and len(ref_pcm_model_sr) > 0:
            ref_sr_t = torch.from_numpy(
                np.asarray(ref_pcm_model_sr, dtype=np.float32).reshape(-1)
            ).to(bundle.device)
            ref_sr_t = ref_sr_t[: int(sr * profile.max_prompt_s)]
        else:
            ref_sr_t = bundle.resample_up(ref16_t.reshape(1, -1)).reshape(-1)

        with torch.no_grad():
            s_ori = bundle.semantic_fn(ref16_t.reshape(1, -1))
            feat = bundle.kaldi.fbank(
                ref16_t.reshape(1, -1),
                num_mel_bins=80,
                dither=0,
                sample_frequency=SAMPLE_RATE_IN,
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            self.style2 = bundle.campplus(feat.unsqueeze(0))
            self.mel2 = bundle.to_mel(ref_sr_t.reshape(1, -1))
            target2_lengths = torch.LongTensor([self.mel2.size(2)]).to(bundle.device)
            self.prompt_condition = bundle.model.length_regulator(
                s_ori, ylens=target2_lengths, n_quantizers=3, f0=None
            )[0]

    # ------------------------------------------------------------------ API

    def convert_block(self, pcm: np.ndarray) -> np.ndarray:
        """Feed 16 kHz mono float32 PCM; returns converted float32 audio."""
        if self._closed:
            raise EngineError("convert_block on a closed session")
        self._pending = np.concatenate(
            [self._pending, np.asarray(pcm, dtype=np.float32).reshape(-1)]
        )
        outs: list[np.ndarray] = []
        while self._pending.size >= self.block_frame_16k:
            block = self._pending[: self.block_frame_16k]
            self._pending = self._pending[self.block_frame_16k :]
            outs.append(self._process_one(block))
        if not outs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(outs) if len(outs) > 1 else outs[0]

    def finish(self) -> np.ndarray:
        """Flush the pending remainder (zero-padded) and the held crossfade."""
        if self._closed:
            return np.zeros(0, dtype=np.float32)
        tail = np.zeros(0, dtype=np.float32)
        if self._pending.size:
            padded = np.pad(self._pending, (0, self.block_frame_16k - self._pending.size))
            self._pending = np.zeros(0, dtype=np.float32)
            tail = self._process_one(padded, final=True)
        if self._carry is not None:
            carry = self._carry.detach().cpu().numpy()
            self._carry = None
            tail = np.concatenate([tail, carry]) if tail.size else carry
        return tail

    def close(self) -> None:
        self._closed = True
        self._pending = np.zeros(0, dtype=np.float32)
        self._carry = None

    # -------------------------------------------------------------- internals

    def _process_one(self, block16k: np.ndarray, *, final: bool = False) -> np.ndarray:
        """Convert exactly one block; returns net-new emitted samples."""
        torch = self._torch
        b = self._b
        device = b.device

        block16k_t = torch.from_numpy(np.ascontiguousarray(block16k)).to(device)

        with torch.no_grad():
            # 1. Slide both windows and append the new block.
            self.input_wav = torch.cat(
                [self.input_wav[self.block_frame :], self._up(block16k_t)]
            )[: self.input_wav.shape[0]]
            self.input_wav_res = torch.cat(
                [self.input_wav_res[self.block_frame_16k :], block16k_t]
            )[: self.input_wav_res.shape[0]]

            # 2. Content features -> length-regulated conditioning.
            s_alt = b.semantic_fn(self.input_wav_res.reshape(1, -1))
            s_alt = s_alt[:, self.ce_dit_frames :]
            target_lengths = torch.LongTensor(
                [
                    int(
                        (
                            self.skip_head
                            + self.return_length
                            + self.skip_tail
                            - self.ce_dit_frames
                        )
                        / MODEL_FRAMES_PER_SECOND
                        * b.sr
                        // b.hop_length
                    )
                ]
            ).to(device)
            cond = b.model.length_regulator(
                s_alt, ylens=target_lengths, n_quantizers=3, f0=None
            )[0]
            cat_condition = torch.cat([self.prompt_condition, cond], dim=1)

            # 3. Flow matching (CFM) + vocoder, under the engine concurrency slot.
            with self._slots:
                try:
                    if b.fp16:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            vc_target = b.model.cfm.inference(
                                cat_condition,
                                torch.LongTensor([cat_condition.size(1)]).to(device),
                                self.mel2,
                                self.style2,
                                None,
                                n_timesteps=self._profile.diffusion_steps,
                                inference_cfg_rate=self._profile.inference_cfg_rate,
                            )
                    else:
                        vc_target = b.model.cfm.inference(
                            cat_condition,
                            torch.LongTensor([cat_condition.size(1)]).to(device),
                            self.mel2,
                            self.style2,
                            None,
                            n_timesteps=self._profile.diffusion_steps,
                            inference_cfg_rate=self._profile.inference_cfg_rate,
                        )
                    vc_target = vc_target[:, :, self.mel2.size(-1) :]
                    vc_wave = b.vocoder_fn(vc_target.float()).squeeze()
                except RuntimeError as exc:  # CUDA OOM / kernel failure
                    raise EngineError(f"seed-vc inference failed: {exc}") from exc

            # 4. Slice this block's portion out of the regenerated window.
            output_len = self.return_length * b.sr // MODEL_FRAMES_PER_SECOND
            tail_len = self.skip_tail * b.sr // MODEL_FRAMES_PER_SECOND
            wave = vc_wave.detach().float().cpu().numpy().reshape(-1)
            need = output_len + tail_len
            if wave.size < need:
                wave = np.pad(wave, (0, need - wave.size))
            out = (
                wave[-need:-tail_len] if tail_len > 0 else wave[-output_len:]
            ).astype(np.float32)

            # 5. Equal-power crossfade over the overlap with the previous chunk.
            ov = self.overlap
            if self._carry is not None:
                carry = self._carry.detach().cpu().numpy().astype(np.float32)
                head = out[:ov] * self._fade_in_np + carry[:ov] * self._fade_out_np
                body = out[ov:] if final else out[ov : self.out_len - ov]
                emitted = np.concatenate([head, body]).astype(np.float32)
                self._carry = (
                    None if final else torch.from_numpy(out[-ov:].copy()).to(device)
                )
            else:
                if final:
                    emitted = out.astype(np.float32)
                    self._carry = None
                else:
                    emitted = out[: self.out_len - ov].astype(np.float32)
                    self._carry = torch.from_numpy(out[-ov:].copy()).to(device)

        return emitted

    def _up(self, block16k_t):
        """Upsample one block from 16 kHz to the model rate (exact length)."""
        up = self._b.resample_up(block16k_t.reshape(1, -1)).reshape(-1)
        if up.shape[0] > self.block_frame:
            up = up[: self.block_frame]
        elif up.shape[0] < self.block_frame:
            up = self._torch.nn.functional.pad(up, (0, self.block_frame - up.shape[0]))
        return up


class SeedVCStreamingEngine:
    """Adapter exposing Seed-VC as a domain ``ConversionEngine``."""

    impl = "seedvc"

    def __init__(self, cfg: EngineSettings) -> None:
        self._cfg = cfg
        self._bundle: SeedVCBundle | None = None
        self._load_lock = threading.Lock()
        self._slots = threading.Semaphore(cfg.max_concurrency)

    @property
    def sr_in(self) -> int:
        return SAMPLE_RATE_IN

    @property
    def sr_out(self) -> int:
        self._ensure_loaded_bundle()
        assert self._bundle is not None
        return self._bundle.sr

    def is_ready(self) -> bool:
        return self._bundle is not None

    def load(self) -> None:
        with self._load_lock:
            if self._bundle is not None:
                return
            logger.info("loading Seed-VC engine (seed-uvit-tat-xlsr-tiny)...")
            t0 = time.perf_counter()
            self._bundle = load_bundle(self._cfg)
            logger.info("engine loaded in %.1fs", time.perf_counter() - t0)

    def _ensure_loaded_bundle(self) -> SeedVCBundle:
        if self._bundle is None:
            raise EngineNotReadyError("engine not loaded (call load() first)")
        return self._bundle

    def new_session(
        self,
        ref_pcm_16k: np.ndarray,
        ref_pcm_model_sr: np.ndarray,
        profile: LatencyProfile,
    ) -> SeedVCConverterSession:
        bundle = self._ensure_loaded_bundle()
        return SeedVCConverterSession(
            bundle, ref_pcm_16k, ref_pcm_model_sr, profile, self._slots
        )

    def preferred_block_samples(self, profile: LatencyProfile) -> int:
        """Input block size (16 kHz samples) quantized to model frames."""
        sr = self.sr_out if self._bundle is not None else 22_050
        zc = sr // MODEL_FRAMES_PER_SECOND
        block_frame = int(round(profile.block_ms / 1000.0 * sr / zc)) * zc
        return 320 * block_frame // zc

    def warmup(self, profile: LatencyProfile) -> dict[str, Any]:
        """Run a few dummy conversions to trigger CUDA kernels & measure RTF."""
        rng = np.random.default_rng(42)
        ref = (rng.standard_normal(16_000) * 0.05).astype(np.float32)
        session = self.new_session(ref, None, profile)
        block = rng.standard_normal(session.block_frame_16k).astype(np.float32) * 0.05
        times: list[float] = []
        try:
            for _ in range(3):
                t0 = time.perf_counter()
                session.convert_block(block)
                times.append(time.perf_counter() - t0)
        finally:
            session.close()
        mean = sum(times) / len(times) if times else 0.0
        stats = {
            "warmup_block_seconds": round(mean, 4),
            "warmup_rtf": round(mean / (profile.block_ms / 1000), 3),
            "device": str(self._ensure_loaded_bundle().device),
        }
        logger.info("engine warmup: %s", stats)
        return stats

    def describe(self) -> dict[str, Any]:
        if self._bundle is None:
            return {"impl": self.impl, "loaded": False}
        return {"impl": self.impl, "loaded": True, **self._bundle.version_info}
