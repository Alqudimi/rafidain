# ============================================================ #
#  الخلية ١٨ — التقييم A/B الذهبي: الأساس مقابل المضبوط        #
#  (جديد في الإصدار ٢ — قلب ادعاء «الأفضل» بالأرقام)           #
#                                                              #
#  نكتب مكتبة تقييم مشتركة + سكربت A/B ثم نشغّلهما:            #
#   • نفس أزواج الاختبار العراقية (معزولة عن التدريب)          #
#   • نفس البذرة العشوائية ونفس خطوات التدفق للنموذجين         #
#   • مقاييس موضوعية: تشابه متحدث CAMPPlus + حفظ محتوى XLSR    #
# ============================================================ #
import json
from pathlib import Path

# ---------- المكتبة المشتركة (يستوردها سكربتا A/B والبنشمارك) ----------
LIB = ROOT / "rafidain_eval_lib.py"
LIB.write_text(r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared evaluation utilities for the Rafidain RunPod notebook (v2).

Faithful ports, no reinvention:
  - stack loading      -> seed-vc inference.py load_models()
  - utterance convert  -> seed-vc inference.py main() (non-f0 path)
  - streaming session  -> rafidain infrastructure/seedvc/engine.py
                          (itself a port of real-time-gui.py)

Run with cwd = seed-vc repository root.
"""
import os, sys, math, argparse
import numpy as np
import torch
import torchaudio
import librosa

SEEDVC_ROOT = os.getcwd()
if SEEDVC_ROOT not in sys.path:
    sys.path.insert(0, SEEDVC_ROOT)

MODEL_FRAMES_PER_SECOND = 50   # seed-vc works on 20 ms frames
SAMPLE_RATE_IN = 16_000
CE_DIT_DIFFERENCE_S = 2.0      # XLSR vs DiT frame offset (upstream constant)


# ------------------------------------------------------------------ loading
def load_stack(checkpoint, config, fp16=True):
    """Load the tiny-XLSR stack exactly like inference.py does."""
    from inference import load_models
    args = argparse.Namespace(checkpoint=str(checkpoint), config=str(config),
                              fp16=bool(fp16), f0_condition=False)
    return load_models(args)


def find_base_ckpt():
    """Locate (or fetch) the published base checkpoint DiT_uvit_tat_xlsr_ema.pth."""
    from hf_utils import load_custom_model_from_hf
    return load_custom_model_from_hf("Plachta/Seed-VC", "DiT_uvit_tat_xlsr_ema.pth", None)


def swap_weights(model, ckpt_path):
    """Hot-swap DiT weights in-place (base -> fine-tuned or back)."""
    from modules.commons import load_checkpoint
    model, _, _, _ = load_checkpoint(model, None, str(ckpt_path),
                                     load_only_params=True, ignore_modules=[],
                                     is_distributed=False)
    for key in model:
        model[key].eval()
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    return model


# ------------------------------------------------------ utterance conversion
@torch.no_grad()
def convert_utterance(model, semantic_fn, vocoder_fn, campplus_model, to_mel,
                      src_path, ref_path, sr, diffusion_steps=25,
                      length_adjust=1.0, inference_cfg_rate=0.7, fp16=True,
                      max_src_sec=None):
    """Port of inference.py main() non-f0 path. Returns (tensor (1,T), out_sr)."""
    from inference import crossfade
    device = next(model.cfm.parameters()).device

    source_audio = librosa.load(src_path, sr=sr)[0]
    ref_audio = librosa.load(ref_path, sr=sr)[0]
    if max_src_sec:
        source_audio = source_audio[: int(sr * max_src_sec)]
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)

    hop_length = 256
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    S_alt = semantic_fn(converted_waves_16k)
    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    mel = to_mel(source_audio.to(device).float())
    mel2 = to_mel(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k, num_mel_bins=80,
                                              dither=0, sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    cond, _, _, _, _ = model.length_regulator(S_alt, ylens=target_lengths,
                                              n_quantizers=3, f0=None)
    prompt_condition, _, _, _, _ = model.length_regulator(S_ori, ylens=target2_lengths,
                                                          n_quantizers=3, f0=None)

    max_source_window = max_context_window - mel2.size(2)
    processed_frames = 0
    generated_wave_chunks = []
    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames: processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(device_type=device.type,
                            dtype=torch.float16 if fp16 else torch.float32):
            vc_target = model.cfm.inference(cat_condition,
                                            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                            mel2, style2, None, diffusion_steps,
                                            inference_cfg_rate=inference_cfg_rate)
            vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = vocoder_fn(vc_target.float()).squeeze()
        vc_wave = vc_wave[None, :]
        if processed_frames == 0:
            if is_last_chunk:
                generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                break
            generated_wave_chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            generated_wave_chunks.append(
                crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len))
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            generated_wave_chunks.append(
                crossfade(previous_chunk.cpu().numpy(),
                          vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len))
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
    vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()
    return vc_wave, sr


# ------------------------------------------------------------------ metrics
def _campplus_embed(campplus_model, wav_np_16k):
    t = torch.from_numpy(np.asarray(wav_np_16k, dtype=np.float32).reshape(1, -1))
    feat = torchaudio.compliance.kaldi.fbank(t, num_mel_bins=80, dither=0,
                                             sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    return campplus_model(feat.unsqueeze(0)).reshape(-1)


def speaker_similarity(campplus_model, wav_a_16k, wav_b_16k):
    """CAMPPlus cosine — timbre closeness to the reference speaker."""
    a = _campplus_embed(campplus_model, wav_a_16k)
    b = _campplus_embed(campplus_model, wav_b_16k)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def _xlsr_embed(semantic_fn, wav_np_16k):
    t = torch.from_numpy(np.asarray(wav_np_16k, dtype=np.float32).reshape(1, -1))
    with torch.no_grad():
        s = semantic_fn(t)
    return s.reshape(-1, s.size(-1)).mean(dim=0)


def content_similarity(semantic_fn, wav_a_16k, wav_b_16k):
    """XLSR mean-pooled cosine — did the words survive the conversion?"""
    a = _xlsr_embed(semantic_fn, wav_a_16k)
    b = _xlsr_embed(semantic_fn, wav_b_16k)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def load_16k(path, max_sec=None):
    y = librosa.load(path, sr=16000)[0]
    return y[: int(16000 * max_sec)] if max_sec else y


# ------------------------------------------------- streaming session (port)
class StreamBenchSession:
    """Port of rafidain infrastructure/seedvc/engine.py SeedVCConverterSession.

    Identical per-block mathematics to the production engine (which itself
    ports real-time-gui.py): sliding window, XLSR content features, length
    regulation, CFM at `diffusion_steps`, HiFT vocoder, equal-power crossfade.
    """

    def __init__(self, stack, profile, ref_path):
        (model, semantic_fn, f0_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args) = stack
        self.model = model
        self.semantic_fn = semantic_fn
        self.vocoder_fn = vocoder_fn
        self.campplus = campplus_model
        self.to_mel = to_mel
        self.kaldi = torchaudio.compliance.kaldi
        self.sr = int(mel_fn_args["sampling_rate"])
        self.hop_length = int(mel_fn_args["hop_size"])
        self.device = next(model.cfm.parameters()).device
        self.fp16 = self.device.type == "cuda"
        self.resample_up = torchaudio.transforms.Resample(16_000, self.sr).to(self.device)
        self.profile = profile

        sr, zc = self.sr, self.sr // MODEL_FRAMES_PER_SECOND

        def q(ms):
            return int(round(ms / 1000.0 * sr / zc)) * zc

        self.block_frame = q(profile["block_ms"])
        self.block_frame_16k = 320 * self.block_frame // zc
        self.crossfade_frame = q(profile["crossfade_ms"])
        self.extra_frame = q(profile["context_left_ms"])
        self.extra_frame_right = q(profile["context_right_ms"])
        self.skip_head = self.extra_frame // zc
        self.skip_tail = self.extra_frame_right // zc
        self.return_length = (self.block_frame + self.crossfade_frame + zc) // zc
        self.ce_dit_frames = int(CE_DIT_DIFFERENCE_S * MODEL_FRAMES_PER_SECOND)

        window = (self.extra_frame + self.crossfade_frame + zc + self.block_frame
                  + self.extra_frame_right)
        self.input_wav = torch.zeros(window, device=self.device, dtype=torch.float32)
        self.input_wav_res = torch.zeros(320 * window // zc, device=self.device,
                                         dtype=torch.float32)

        self.out_len = self.return_length * sr // MODEL_FRAMES_PER_SECOND
        self.overlap = self.out_len - self.block_frame
        assert self.overlap > 0 and self.block_frame > self.overlap, "block_ms too small"
        ramp = torch.sin(0.5 * float(np.pi) * torch.linspace(
            0.0, 1.0, steps=self.overlap, device=self.device)) ** 2
        self.fade_in = ramp.float()
        self._fade_in_np = ramp.detach().cpu().numpy().astype(np.float32)
        self._fade_out_np = 1.0 - self._fade_in_np
        self._carry = None
        self._pending = np.zeros(0, dtype=np.float32)

        # ---- condition on the reference voice (same as production engine) ----
        max_ref = int(16000 * profile["max_prompt_s"])
        ref16 = load_16k(ref_path)[:max_ref]
        assert ref16.size >= 16000, "reference too short"
        ref16_t = torch.from_numpy(ref16).to(self.device)
        ref_sr_t = self.resample_up(ref16_t.reshape(1, -1)).reshape(-1)

        with torch.no_grad():
            s_ori = self.semantic_fn(ref16_t.reshape(1, -1))
            feat = self.kaldi.fbank(ref16_t.reshape(1, -1), num_mel_bins=80,
                                    dither=0, sample_frequency=SAMPLE_RATE_IN)
            feat = feat - feat.mean(dim=0, keepdim=True)
            self.style2 = self.campplus(feat.unsqueeze(0))
            self.mel2 = self.to_mel(ref_sr_t.reshape(1, -1))
            target2_lengths = torch.LongTensor([self.mel2.size(2)]).to(self.device)
            self.prompt_condition = self.model.length_regulator(
                s_ori, ylens=target2_lengths, n_quantizers=3, f0=None)[0]

    def convert_block(self, pcm):
        self._pending = np.concatenate(
            [self._pending, np.asarray(pcm, dtype=np.float32).reshape(-1)])
        outs = []
        while self._pending.size >= self.block_frame_16k:
            block = self._pending[: self.block_frame_16k]
            self._pending = self._pending[self.block_frame_16k:]
            outs.append(self._process_one(block))
        if not outs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(outs) if len(outs) > 1 else outs[0]

    def finish(self):
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

    def _process_one(self, block16k, final=False):
        device = self.device
        block16k_t = torch.from_numpy(np.ascontiguousarray(block16k)).to(device)
        with torch.no_grad():
            self.input_wav = torch.cat(
                [self.input_wav[self.block_frame:], self._up(block16k_t)]
            )[: self.input_wav.shape[0]]
            self.input_wav_res = torch.cat(
                [self.input_wav_res[self.block_frame_16k:], block16k_t]
            )[: self.input_wav_res.shape[0]]

            s_alt = self.semantic_fn(self.input_wav_res.reshape(1, -1))
            s_alt = s_alt[:, self.ce_dit_frames:]
            target_lengths = torch.LongTensor(
                [int((self.skip_head + self.return_length + self.skip_tail
                      - self.ce_dit_frames) / MODEL_FRAMES_PER_SECOND
                     * self.sr // self.hop_length)]).to(device)
            cond = self.model.length_regulator(s_alt, ylens=target_lengths,
                                               n_quantizers=3, f0=None)[0]
            cat_condition = torch.cat([self.prompt_condition, cond], dim=1)

            if self.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    vc_target = self.model.cfm.inference(
                        cat_condition,
                        torch.LongTensor([cat_condition.size(1)]).to(device),
                        self.mel2, self.style2, None,
                        self.profile["diffusion_steps"],
                        inference_cfg_rate=self.profile["inference_cfg_rate"])
            else:
                vc_target = self.model.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(device),
                    self.mel2, self.style2, None,
                    self.profile["diffusion_steps"],
                    inference_cfg_rate=self.profile["inference_cfg_rate"])
            vc_target = vc_target[:, :, self.mel2.size(-1):]
            vc_wave = self.vocoder_fn(vc_target.float()).squeeze()

            output_len = self.return_length * self.sr // MODEL_FRAMES_PER_SECOND
            tail_len = self.skip_tail * self.sr // MODEL_FRAMES_PER_SECOND
            wave = vc_wave.detach().float().cpu().numpy().reshape(-1)
            need = output_len + tail_len
            if wave.size < need:
                wave = np.pad(wave, (0, need - wave.size))
            out = (wave[-need:-tail_len] if tail_len > 0 else wave[-output_len:]).astype(np.float32)

            ov = self.overlap
            if self._carry is not None:
                carry = self._carry.detach().cpu().numpy().astype(np.float32)
                head = out[:ov] * self._fade_in_np + carry[:ov] * self._fade_out_np
                body = out[ov:] if final else out[ov: self.out_len - ov]
                emitted = np.concatenate([head, body]).astype(np.float32)
                self._carry = None if final else torch.from_numpy(out[-ov:].copy()).to(device)
            else:
                if final:
                    emitted = out.astype(np.float32)
                    self._carry = None
                else:
                    emitted = out[: self.out_len - ov].astype(np.float32)
                    self._carry = torch.from_numpy(out[-ov:].copy()).to(device)
        return emitted

    def _up(self, block16k_t):
        up = self.resample_up(block16k_t.reshape(1, -1)).reshape(-1)
        if up.shape[0] > self.block_frame:
            up = up[: self.block_frame]
        elif up.shape[0] < self.block_frame:
            up = torch.nn.functional.pad(up, (0, self.block_frame - up.shape[0]))
        return up
''', encoding="utf-8")
print(f"[مكتبة التقييم] كُتبت: {LIB}")

# ---------- سكربت A/B: الأساس مقابل المضبوط ----------
AB = ROOT / "ab_eval.py"
AB.write_text(r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B evaluation: published base Seed-VC vs Rafidain Iraqi fine-tune.

Fair by construction: identical held-out Iraqi pairs, identical random seed
per pair (same CFM noise), identical diffusion steps. Reports CAMPPlus
speaker similarity + XLSR content preservation per pair and averaged.
"""
import os, sys, json, glob, time, argparse, random
os.environ.setdefault("HF_HUB_CACHE", "./checkpoints/hf_cache")
sys.path.insert(0, os.environ.get("NB_ROOT", "."))

import numpy as np
import torch
import torchaudio
from rafidain_eval_lib import (load_stack, find_base_ckpt, swap_weights,
                               convert_utterance, speaker_similarity,
                               content_similarity, load_16k)

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")


def list_audio(d):
    out = []
    for root, _, files in os.walk(d):
        for f in sorted(files):
            if f.lower().endswith(AUDIO_EXTS):
                out.append(os.path.join(root, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--fallback-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--diffusion-steps", type=int, default=25)
    args = ap.parse_args()

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    clips = list_audio(args.test_dir)
    in_train = False
    if len(clips) < 2 * args.pairs and args.fallback_dir:
        extra = [p for p in list_audio(args.fallback_dir)
                 if os.path.basename(p) not in {os.path.basename(c) for c in clips}]
        clips = clips + extra
        in_train = True
    assert len(clips) >= 2 * args.pairs, (
        f"need {2*args.pairs} clips, found {len(clips)}")
    if in_train:
        print("[WARN] test set too small - falling back to TRAINING clips "
              "(evaluation less honest; prefer the holdout cell)")

    random.shuffle(clips)
    sources = clips[: args.pairs]
    refs = clips[args.pairs: 2 * args.pairs]

    os.makedirs(args.out_dir, exist_ok=True)

    print("[load] base stack (DiT_uvit_tat_xlsr_ema) ...")
    base_ckpt = find_base_ckpt()
    t0 = time.time()
    stack = load_stack(base_ckpt, args.config, fp16=True)
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args = stack
    sr = int(mel_fn_args["sampling_rate"])
    print(f"[load] ready in {time.time()-t0:.0f}s (sr={sr})")

    results = {"config": {"diffusion_steps": args.diffusion_steps,
                          "pairs": args.pairs, "sr": sr,
                          "held_out": not in_train},
               "pairs": [], "summary": {}}

    def eval_model(tag):
        rows, fails = [], []
        for i, (src, ref) in enumerate(zip(sources, refs)):
            try:
                torch.manual_seed(1000 + i)      # same CFM noise for both models
                wav, out_sr = convert_utterance(
                    model, semantic_fn, vocoder_fn, campplus_model, to_mel,
                    src, ref, sr, diffusion_steps=args.diffusion_steps,
                    fp16=True, max_src_sec=12.0)
                out_name = f"{tag}_pair{i:02d}.wav"
                torchaudio.save(os.path.join(args.out_dir, out_name), wav.cpu(), out_sr)
                conv16 = wav.reshape(-1).cpu().numpy()
                conv16 = torchaudio.functional.resample(
                    torch.from_numpy(conv16), out_sr, 16000).numpy()
                sim_spk = speaker_similarity(campplus_model, conv16, load_16k(ref, 25))
                sim_cnt = content_similarity(semantic_fn, conv16, load_16k(src, 12))
                rows.append({"pair": i, "source": os.path.basename(src),
                             "reference": os.path.basename(ref),
                             "speaker_sim": round(sim_spk, 4),
                             "content_sim": round(sim_cnt, 4)})
                print(f"  [{tag}] pair {i}: speaker={sim_spk:.4f}  content={sim_cnt:.4f}")
            except Exception as exc:   # مقطع تالف لا يوقف التقييم كله
                fails.append({"pair": i, "error": str(exc)[:200]})
                print(f"  [{tag}] pair {i}: FAILED ({exc})")
        if fails:
            results.setdefault("failures", {})[tag] = fails
        assert rows, f"all pairs failed for {tag} — check audio files / logs above"
        return rows

    print("\n[eval] BASE model ...")
    results["base"] = eval_model("base")

    print("\n[swap] loading fine-tuned weights ...")
    swap_weights(model, args.ft_ckpt)

    print("\n[eval] FINE-TUNED model ...")
    results["ft"] = eval_model("ft")

    # ---- aggregate (على الأزواج المشتركة الناجحة في النموذجين — مقارنة عادلة) ----
    common = sorted({r["pair"] for r in results["base"]} &
                    {r["pair"] for r in results["ft"]})
    if len(common) < len(sources):
        print(f"[note] computing means over {len(common)} common successful pairs")
    b_spk = [r["speaker_sim"] for r in results["base"] if r["pair"] in common]
    f_spk = [r["speaker_sim"] for r in results["ft"] if r["pair"] in common]
    b_cnt = [r["content_sim"] for r in results["base"] if r["pair"] in common]
    f_cnt = [r["content_sim"] for r in results["ft"] if r["pair"] in common]
    base_spk = float(np.mean(b_spk))
    ft_spk = float(np.mean(f_spk))
    base_cnt = float(np.mean(b_cnt))
    ft_cnt = float(np.mean(f_cnt))
    results["summary"] = {
        "base_speaker_sim_mean": round(base_spk, 4),
        "ft_speaker_sim_mean": round(ft_spk, 4),
        "speaker_sim_delta": round(ft_spk - base_spk, 4),
        "base_content_sim_mean": round(base_cnt, 4),
        "ft_content_sim_mean": round(ft_cnt, 4),
        "content_sim_delta": round(ft_cnt - base_cnt, 4),
    }
    s = results["summary"]
    verdict = "IMPROVED" if (s["speaker_sim_delta"] > 0
                             and s["content_sim_delta"] > -0.02) else "NEUTRAL/REGRESSION"
    results["summary"]["verdict"] = verdict

    out_json = os.path.join(args.out_dir, "ab_eval.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("A/B SUMMARY  (CAMPPlus speaker sim | XLSR content sim)")
    print("=" * 70)
    print(f"  base model : {base_spk:.4f} | {base_cnt:.4f}")
    print(f"  fine-tuned : {ft_spk:.4f} | {ft_cnt:.4f}")
    print(f"  delta      : {s['speaker_sim_delta']:+.4f} | {s['content_sim_delta']:+.4f}")
    print(f"  verdict    : {verdict}")
    print(f"  (details)  : {out_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
''', encoding="utf-8")
print(f"[سكربت A/B] كُتب: {AB}")

# ---------- تشغيل التقييم ----------
FT = RUN_DIR / "ft_model.pth"
assert FT.exists(), "لا يوجد ft_model.pth — أكمل التدريب أولاً (الخلية 14)"
CFG_TINY = SEEDVC / "configs" / "presets" / "config_dit_mel_seed_uvit_xlsr_tiny.yml"
AB_DIR = RUN_DIR / "ab_eval"

env = dict(ENV)
env["NB_ROOT"] = str(ROOT)
sh([VENV_PY, AB,
    "--ft-ckpt", FT,
    "--config", CFG_TINY,
    "--test-dir", TEST_DIR,
    "--fallback-dir", DATA_DIR,
    "--out-dir", AB_DIR,
    "--pairs", int(CFG["EVAL_PAIRS"]),
    "--diffusion-steps", int(CFG["EVAL_STEPS"])],
   cwd=SEEDVC, env=env)

# ---------- عرض النتائج: جدول + استماع مقارن ----------
from IPython.display import Audio, display

rep = json.loads((AB_DIR / "ab_eval.json").read_text(encoding="utf-8"))
s = rep["summary"]
print(f"""
تشابه المتحدث (CAMPPlus):  الأساس {s['base_speaker_sim_mean']:.4f}  ←  المضبوط {s['ft_speaker_sim_mean']:.4f}   (فرق {s['speaker_sim_delta']:+.4f})
الحفاظ على المحتوى (XLSR):  الأساس {s['base_content_sim_mean']:.4f}  ←  المضبوط {s['ft_content_sim_mean']:.4f}   (فرق {s['content_sim_delta']:+.4f})
الفيصل: {s['verdict']}
{'⚠ المقاطع من بيانات التدريب (لا يوجد عزل) — النتيجة متفائلة نسبياً؛ شغّل خلية العزل 12 وأعد التقييم' if not rep['config']['held_out'] else '✔ التقييم على مقاطع معزولة لم يدخلها التدريب'}
""")
print("استمع للمقارنة (الزوج الأول): المصدر ← الأساس ← المضبوط، مع المرجع العراقي:")
pair0_src = next((p["source"] for p in rep["ft"]), None)
if pair0_src:
    print(f"- المصدر: {pair0_src}")
for tag, label in (("base", "ناتج الأساس"), ("ft", "ناتج المضبوط")):
    f = AB_DIR / f"{tag}_pair00.wav"
    if f.exists():
        print(f"- {label}:")
        display(Audio(filename=str(f)))
ref_name = rep["ft"][0]["reference"] if rep["ft"] else None
ref_f = None
if ref_name:
    for d in (TEST_DIR, DATA_DIR / "yodas", DATA_DIR / "youssef", DATA_DIR / "safaa", DATA_DIR / "custom"):
        cand = d / ref_name
        if cand.exists():
            ref_f = cand
            break
if ref_f:
    print(f"- المرجع العراقي: {ref_f.name}")
    display(Audio(filename=str(ref_f)))
