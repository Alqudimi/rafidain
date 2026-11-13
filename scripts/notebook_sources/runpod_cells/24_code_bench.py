# ============================================================ #
#  الخلية ١٩ — بنشمارك الاستنساخ اللحظي (جديد في الإصدار ٢)     #
#                                                              #
#  يقيس على كرتك الفعلي، بالنموذج المضبوط:                      #
#   أ) RTF على مستوى المقطع عند 10/16/30 خطوة تدفق             #
#   ب) **البث الإنتاجي الحقيقي**: بروفايل balanced               #
#      (كتل 180ms + crossfade 40ms + سياق 20ms + 10 خطوات)     #
#      بنفس منطج محرك رافِدَين الإنتاجي — يعطي p50/p95          #
#      وزمن الوصول الشامل المقدَّر                              #
# ============================================================ #
import json
from pathlib import Path

BENCH = ROOT / "bench_stream.py"
BENCH.write_text(r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Latency benchmark for the fine-tuned Iraqi model.

Part A: utterance-level RTF at diffusion steps {10, 16, 30}.
Part B: production streaming simulation with the `balanced` profile
        (block 180ms / crossfade 40ms / right ctx 20ms / 10 steps) using
        StreamBenchSession - a faithful port of rafidain's production
        engine (which ports seed-vc's real-time-gui.py).

RTF < 1.0 means faster than real time. The published measurement for this
model family on the balanced profile is ~430 ms end-to-end on a decent GPU.
"""
import os, sys, json, time, argparse, random
os.environ.setdefault("HF_HUB_CACHE", "./checkpoints/hf_cache")
sys.path.insert(0, os.environ.get("NB_ROOT", "."))

import numpy as np
import torch
import torchaudio
from rafidain_eval_lib import load_stack, convert_utterance, load_16k, StreamBenchSession

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
    args = ap.parse_args()

    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)

    clips = list_audio(args.test_dir)
    if len(clips) < 2 and args.fallback_dir:
        clips = clips + list_audio(args.fallback_dir)
    assert len(clips) >= 2, "need at least 2 clips (source + reference)"

    # جرب المقاطع بالترتيب حتى ينجح تحميل واحد (مقطع تالف لا يكسر البنشمارك)
    import librosa as _lib
    def _try_load(path, sr=16000, max_sec=None):
        try:
            y = _lib.load(path, sr=sr)[0]
            if max_sec:
                y = y[: int(sr * max_sec)]
            return y if y.size > sr else None   # تجاهل الأقصر من ثانية
        except Exception:
            return None

    src = ref = None
    for c in clips:
        if src is None and _try_load(c) is not None:
            src = c
        elif ref is None and _try_load(c) is not None:
            ref = c
        if src is not None and ref is not None:
            break
    assert src is not None and ref is not None, "no loadable clips found"
    print(f"[bench] source: {os.path.basename(src)}")
    print(f"[bench] reference: {os.path.basename(ref)}")

    print("\n[load] fine-tuned stack ...")
    t0 = time.time()
    stack = load_stack(args.ft_ckpt, args.config, fp16=True)
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args = stack
    sr = int(mel_fn_args["sampling_rate"])
    print(f"[load] ready in {time.time()-t0:.0f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    report = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "sr": sr, "utterance_rtf": {}, "streaming": {}}

    # ---- Part A: utterance RTF ----
    print("\n[A] utterance-level RTF (convert ~10s clip, model already loaded):")
    for steps in (10, 16, 30):
        # warmup once for kernel autotuning
        convert_utterance(model, semantic_fn, vocoder_fn, campplus_model, to_mel,
                          src, ref, sr, diffusion_steps=steps, fp16=True, max_src_sec=10)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        wav, out_sr = convert_utterance(model, semantic_fn, vocoder_fn, campplus_model,
                                        to_mel, src, ref, sr, diffusion_steps=steps,
                                        fp16=True, max_src_sec=10)
        dur = wav.size(-1) / out_sr
        wall = time.perf_counter() - t0
        rtf = wall / dur
        report["utterance_rtf"][str(steps)] = {"wall_s": round(wall, 3),
                                               "audio_s": round(dur, 2),
                                               "rtf": round(rtf, 4)}
        print(f"    steps={steps:>2}: {wall:.2f}s wall for {dur:.1f}s audio  ->  RTF {rtf:.3f}"
              + ("  (أسرع من الزمن الحقيقي)" if rtf < 1 else "  (أبطأ من الزمن الحقيقي!)"))

    # ---- Part B: streaming with production profile ----
    profiles = {
        "balanced": dict(block_ms=180, crossfade_ms=40, context_left_ms=2500,
                         context_right_ms=20, diffusion_steps=10,
                         inference_cfg_rate=0.7, max_prompt_s=3.0),
        "ultra_low": dict(block_ms=120, crossfade_ms=40, context_left_ms=2000,
                          context_right_ms=20, diffusion_steps=4,
                          inference_cfg_rate=0.0, max_prompt_s=3.0),
    }
    src16 = load_16k(src, 8.0)   # 8 ثوانٍ من الكلام العراقي الحقيقي
    print("\n[B] streaming simulation (production engine math):")
    for name, prof in profiles.items():
        sess = StreamBenchSession(stack, prof, ref)
        block = sess.block_frame_16k
        times = []
        emitted_all = []
        pos = 0
        n_blocks = 0
        while pos + block <= src16.size:
            pcm = src16[pos: pos + block]
            t0 = time.perf_counter()
            out = sess.convert_block(pcm)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            emitted_all.append(out)
            if n_blocks >= 3:          # skip first blocks (CUDA warmup)
                times.append(dt)
            n_blocks += 1
            pos += block
        emitted_all.append(sess.finish())
        if not times:
            print(f"    [{name}] المصدر أقصر من 3 كتل — تخطّي")
            continue
        if name == "balanced" and emitted_all:
            # احتفظ بصوت البث للاستماع — دليل ملموس أن مسار الإنتاج يعمل بالنقطة المضبوطة
            import numpy as _np
            full = _np.concatenate([e for e in emitted_all if e.size])
            torchaudio.save(os.path.join(args.out_dir, "stream_balanced_out.wav"),
                            torch.from_numpy(full).reshape(1, -1).float(), sess.sr)
            print(f"    [{name}] الصوت المتدفق حُفظ: stream_balanced_out.wav "
                  f"({full.size/sess.sr:.1f}s)")
        times_np = np.array(times)
        p50 = float(np.percentile(times_np, 50)) * 1000
        p95 = float(np.percentile(times_np, 95)) * 1000
        mean = float(times_np.mean()) * 1000
        rtf = float(times_np.mean()) / (prof["block_ms"] / 1000)
        algo = prof["block_ms"] + prof["crossfade_ms"] + prof["context_right_ms"]
        e2e = algo + mean
        report["streaming"][name] = {
            "block_ms": prof["block_ms"], "diffusion_steps": prof["diffusion_steps"],
            "n_blocks_measured": int(times_np.size),
            "block_time_p50_ms": round(p50, 1), "block_time_p95_ms": round(p95, 1),
            "block_time_mean_ms": round(mean, 1),
            "algorithmic_delay_ms": algo,
            "estimated_e2e_ms": round(e2e, 1),
            "rtf": round(rtf, 3),
        }
        print(f"    [{name}] كتلة {prof['block_ms']}ms × {prof['diffusion_steps']} خطوة:")
        print(f"        زمن معالجة الكتلة: p50 {p50:.0f}ms | p95 {p95:.0f}ms | RTF {rtf:.2f}")
        print(f"        زمن الوصول الشامل المقدَّر (كتلة+crossfade+سياق+استدلال): {e2e:.0f}ms"
              + ("  ← داخل ميزانية المكالمات" if e2e < 500 else "  ← فوق الميزانية"))

    out_json = os.path.join(args.out_dir, "bench_stream.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n[bench] report: {out_json}")


if __name__ == "__main__":
    main()
''', encoding="utf-8")
print(f"[سكربت البنشمارك] كُتب: {BENCH}")

# ---------- تشغيل البنشمارك ----------
FT = RUN_DIR / "ft_model.pth"
assert FT.exists(), "لا يوجد ft_model.pth — أكمل التدريب أولاً (الخلية 14)"
CFG_TINY = SEEDVC / "configs" / "presets" / "config_dit_mel_seed_uvit_xlsr_tiny.yml"
BENCH_DIR = RUN_DIR / "bench"

env = dict(ENV)
env["NB_ROOT"] = str(ROOT)
sh([VENV_PY, BENCH,
    "--ft-ckpt", FT,
    "--config", CFG_TINY,
    "--test-dir", TEST_DIR,
    "--fallback-dir", DATA_DIR,
    "--out-dir", BENCH_DIR],
   cwd=SEEDVC, env=env)

rep = json.loads((BENCH_DIR / "bench_stream.json").read_text(encoding="utf-8"))
b = rep["streaming"].get("balanced", {})
if b:
    print(f"""
خلاصة «الاستنساخ اللحظي» على {rep['gpu']}:
  • زمن معالجة كتلة البث (p95): {b['block_time_p95_ms']:.0f}ms — أي أن الصوت يخرج أسرع من دخوله بمعامل {b['rtf']:.2f}x
  • زمن الوصول الشامل المقدَّر: {b['estimated_e2e_ms']:.0f}ms (المرجع المنشور للنموذج نفسه: ~430ms)
  • RTF على مستوى المقطع (10 خطوات): {rep['utterance_rtf'].get('10', {}).get('rtf', '?')}
هذه أرقام جهازك الفعلي بالنموذج المضبوط — احتفظ بها (ستُرفع تلقائياً مع حزمة النموذج).""")

# استمع لناتج البث اللحظي الحقيقي (بروفايل balanced — نفس مسار الإنتاج)
from IPython.display import Audio, display
streamed = BENCH_DIR / "stream_balanced_out.wav"
if streamed.exists():
    print("\nاستمع لناتج البث اللحظي (كتل 180ms متلاحقة عبر محرك الإنتاج — بالنقطة المضبوطة):")
    display(Audio(filename=str(streamed)))
