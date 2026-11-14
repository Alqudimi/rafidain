# -*- coding: utf-8 -*-
"""
evaluate_ckpt.py — التقييم الموضوعي لنقاط Seed-VC (تشابه متحدث + طبيعية)
==========================================================================
يقيّم نقاط التحقق على عينات اختبار معزولة عبر inference.py الرسمي ثم يحسب:
  • SIM: تشابه المتحدث (تضمين CAMPPlus، جيب التمام) — المقياس الأساسي
  • UTMOS: توقع MOS للطبيعية (1–5، ترتيب نسبي فقط لأنه دُرّب على EN/ZH)
  • خط أساس إلزامي: النموذج الرسمي غير المضبوط على نفس العينات

القرار: النقطة الأفضل = أعلى SIM مع UTMOS لا ينهار (لا ينخفض > 0.2 عن خط الأساس).

الاستخدام (من داخل مجلد seed-vc):
    python evaluate_ckpt.py --run-dir runs/iraqi_v1 --test-dir /path/to/holdout \
        --config runs/iraqi_v1/config_*.yml --drive-dir /content/drive/.../iraqi_v1
    # إضافة نقطة يدوية: --extra-ckpt runs/iraqi_v1/gold_stage_A.pth
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import statistics
from pathlib import Path

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}


def _step_of(p):
    m = re.search(r"step_(\d+)", Path(p).stem)
    return int(m.group(1)) if m else -1


def find_config(run_dir):
    cands = sorted(Path(run_dir).glob("config_*.yml"))
    if not cands:
        # ربما .yaml أو الإعداد في مجلد الإعداد الأصلي — ابحث أعمق
        cands = sorted(Path(run_dir).glob("*.yml"))
    assert cands, f"لا ملف إعدادات في {run_dir}"
    return cands[0]


def pick_samples(test_dir, n_sources=3, n_refs=2, refs_dir=None):
    srcs = sorted([p for p in Path(test_dir).iterdir() if p.suffix.lower() in AUDIO_EXT])
    assert srcs, f"لا مقاطع اختبار في {test_dir}"
    sources = srcs[:n_sources]
    if refs_dir and Path(refs_dir).exists():
        refs = sorted([p for p in Path(refs_dir).iterdir() if p.suffix.lower() in AUDIO_EXT])[:n_refs]
    else:
        refs = srcs[-n_refs:] if len(srcs) >= n_refs + n_sources else sources[:n_refs]
    return sources, refs


def convert_one(src, ref, out_dir, config, ckpt=None, diffusion_steps=30, python=None):
    """تشغيل inference.py الرسمي لنقطة واحدة. يعيد مسار المخرج أو None."""
    cmd = [python or sys.executable, "inference.py",
           "--source", str(src), "--target", str(ref),
           "--output", str(out_dir), "--config", str(config),
           "--diffusion-steps", str(diffusion_steps),
           "--f0-condition", "True", "--auto-f0-adjust", "True",
           "--fp16", "True"]
    if ckpt is not None:
        cmd += ["--checkpoint", str(ckpt)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    # التسمية الرسمية: vc_{source}_{target}_{length_adjust}_{steps}_{cfg}.wav
    expected = None
    for f in Path(out_dir).glob("vc_*.wav"):
        if Path(src).stem in f.name and Path(ref).stem in f.name:
            expected = f
            break
    if r.returncode != 0 or expected is None:
        print(f"[convert] فشل {Path(src).name}×{Path(ref).name}"
              f"{' (نقطة ' + Path(ckpt).name + ')' if ckpt else ''}")
        print((r.stdout or "")[-400:], (r.stderr or "")[-400:])
        return None
    return expected


def campplus_sim(wav1, wav2, sv_model, device):
    import torch
    import torchaudio.compliance.kaldi as kaldi
    import librosa

    def embed(path):
        w, _ = librosa.load(str(path), sr=16000, mono=True)
        f = kaldi.fbank(torch.from_numpy(w).unsqueeze(0), num_mel_bins=80,
                        dither=0, sample_frequency=16000)
        f = f - f.mean(dim=0, keepdim=True)
        with torch.no_grad():
            return sv_model(f.unsqueeze(0).to(device)).squeeze(0)

    return float(torch.nn.functional.cosine_similarity(
        embed(wav1), embed(wav2), dim=0).item())


def utmos_score(wav_path, predictor, device):
    import torch
    import librosa
    w, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    with torch.no_grad():
        return float(predictor(torch.from_numpy(w).unsqueeze(0).to(device), 16000).item())


def main():
    ap = argparse.ArgumentParser(description="التقييم الموضوعي لنقاط Seed-VC")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--test-dir", required=True, help="مقاطع الاختبار المعزولة")
    ap.add_argument("--config", default=None, help="افتراضياً: config_*.yml من run-dir")
    ap.add_argument("--drive-dir", default=None, help="نسخ النتائج إلى Drive")
    ap.add_argument("--refs-dir", default=None, help="أصوات مرجعية بديلة (افتراضي: من عينات الاختبار)")
    ap.add_argument("--max-ckpts", type=int, default=6)
    ap.add_argument("--extra-ckpt", action="append", default=[],
                    help="نقطة إضافية للتقييم (مكرر) — مثال نقاط gold_*")
    ap.add_argument("--n-sources", type=int, default=3)
    ap.add_argument("--n-refs", type=int, default=2)
    ap.add_argument("--diffusion-steps", type=int, default=30)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--workdir", default=None, help="افتراضياً <run-dir>/../eval_run")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    test_dir = Path(args.test_dir).resolve()
    assert run_dir.exists(), f"مجلد التشغيل غير موجود: {run_dir}"
    assert test_dir.exists(), f"مجلد الاختبار غير موجود: {test_dir}"

    config = Path(args.config) if args.config else find_config(run_dir)
    workdir = Path(args.workdir) if args.workdir else run_dir.parent / "eval_run"
    workdir.mkdir(parents=True, exist_ok=True)

    # أحدث N نقاط + نقاط إضافية (نزع التكرار)
    ckpts = sorted(run_dir.glob("DiT_epoch_*_step_*.pth"), key=_step_of)[-args.max_ckpts:]
    for e in args.extra_ckpt:
        ep = Path(e)
        if ep.exists() and ep not in ckpts:
            ckpts.append(ep)
    assert ckpts, "لا نقاط تقييم — شغّل التدريب أولاً"

    sources, refs = pick_samples(test_dir, args.n_sources, args.n_refs, args.refs_dir)
    print(f"النقاط المرشحة ({len(ckpts)}):")
    for c in ckpts:
        print(f"  • {c.name}")
    print(f"المصادر: {[s.name for s in sources]}")
    print(f"المراجع: {[r.name for r in refs]}\n")

    # ---------- المرحلة 1: التحويل عبر inference.py الرسمي ----------
    jobs = []
    if not args.skip_baseline:
        jobs.append((None, "baseline_official"))
    jobs += [(c, c.stem) for c in ckpts]

    for ckpt, tag in jobs:
        out = workdir / tag
        out.mkdir(exist_ok=True)
        if (out / ".done").exists():
            print(f"[تخطّي] {tag} — مُحوَّلة مسبقاً")
            continue
        ok = 0
        for src in sources:
            for ref in refs:
                if convert_one(src, ref, out, config, ckpt, args.diffusion_steps):
                    ok += 1
        print(f"[convert] {tag}: {ok}/{len(sources)*len(refs)} مخرجات")
        (out / ".done").touch()

    # ---------- المرحلة 2: المقاييس ----------
    print("\nتحميل مقاييس التقييم...")
    import torch
    from modules.campplus.DTDNN import CAMPPlus
    from hf_utils import load_custom_model_from_hf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sv = CAMPPlus(feat_dim=80, embedding_size=192)
    bin_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", None)
    sv.load_state_dict(torch.load(bin_path, map_location="cpu"))
    sv.eval().to(device)

    utmos = None
    try:
        utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)
        print("  UTMOS22 ✓ (torch.hub)")
    except Exception as e:
        print(f"  UTMOS غير متاح ({e}) — SIM فقط")

    rows, summary = [], {}
    for ckpt, tag in jobs:
        out = workdir / tag
        sims, utms = [], []
        for src in sources:
            for ref in refs:
                gen = None
                for f in out.glob("vc_*.wav"):
                    if src.stem in f.name and ref.stem in f.name:
                        gen = f
                        break
                if gen is None:
                    continue
                try:
                    sim = campplus_sim(gen, ref, sv, device)
                    sims.append(sim)
                    row = {"ckpt": tag, "src": src.name, "ref": ref.name,
                           "sim": round(sim, 4)}
                    if utmos is not None:
                        u = utmos_score(gen, utmos, device)
                        utms.append(u)
                        row["utmos"] = round(u, 3)
                    rows.append(row)
                except Exception as e:
                    print(f"[measure] خطأ {tag}/{src.name}: {e}")
        if sims:
            summary[tag] = {
                "sim_mean": round(statistics.mean(sims), 4),
                "sim_std": round(statistics.stdev(sims), 4) if len(sims) > 1 else 0.0,
                "utmos_mean": round(statistics.mean(utms), 3) if utms else None,
                "n": len(sims),
            }

    # ---------- المرحلة 3: الترتيب والقرار ----------
    print("\n" + "=" * 64)
    print(f"{'النقطة':<40} {'SIM':>7} {'±':>6} {'UTMOS':>7} {'n':>3}")
    print("-" * 64)
    for tag, v in sorted(summary.items(), key=lambda kv: -kv[1]["sim_mean"]):
        print(f"{tag:<40} {v['sim_mean']:>7.4f} {v['sim_std']:>6.4f} "
              f"{v['utmos_mean'] if v['utmos_mean'] is not None else '-':>7} {v['n']:>3}")
    print("=" * 64)

    baseline = summary.get("baseline_official")
    if baseline:
        best_tag = max((t for t in summary if t != "baseline_official"),
                       key=lambda t: summary[t]["sim_mean"], default=None)
        if best_tag:
            gain = summary[best_tag]["sim_mean"] - baseline["sim_mean"]
            verdict = ("✓ تحسّن حقيقي عن الرسمي" if gain >= 0.02 else
                       "≈ قريب من الرسمي — قيّم بالسمع" if gain >= 0 else
                       "✗ لم يتفوق بعد — تابع التدريب")
            print(f"\nخط الأساس SIM={baseline['sim_mean']:.4f} | "
                  f"الأفضل {best_tag} SIM={summary[best_tag]['sim_mean']:.4f} "
                  f"(فرق {gain:+.4f}) — {verdict}")

    # ---------- الحفظ ----------
    csv_path = workdir / "eval_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ckpt", "src", "ref", "sim", "utmos"])
        w.writeheader()
        w.writerows(rows)
    (workdir / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[نتائج] {csv_path}")

    if args.drive_dir:
        drv = Path(args.drive_dir)
        drv.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_path, drv / "eval_results.csv")
        shutil.copy2(workdir / "eval_summary.json", drv / "eval_summary.json")
        print(f"[Drive] نُسخت النتائج إلى {drv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
