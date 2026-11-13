# ============================================================ #
#  الخلية ١٧ — التحقق الأولي: استدلال حقيقي بالنقطة المضبوطة    #
#  (مصدر من أمثلة Seed-VC + مرجع عراقي من بياناتك -> تشغيل)    #
#  فشل هذه الخلية غير قاتل: الأوزان تصدَّر على أي حال           #
# ============================================================ #
from IPython.display import Audio, display
from pathlib import Path
import shutil

FT = RUN_DIR / "ft_model.pth"
assert FT.exists(), "لا يوجد ft_model.pth — شغّل خلية الإطلاق (14) وراقب حتى الاكتمال"
CFG_TINY = SEEDVC / "configs" / "presets" / "config_dit_mel_seed_uvit_xlsr_tiny.yml"

sh([VENV_PY, PIPE, "verify",
    "--repo", SEEDVC,
    "--run-dir", RUN_DIR,
    "--ckpt", FT,
    "--config", CFG_TINY,
    "--data-dir", DATA_DIR], check=False)   # فشل verify ليس قاتلاً — الأوزان تصدَّر على أي حال

outs = sorted((RUN_DIR / "verify_out").glob("vc_*.wav"))
if not outs:
    # احتياط (جديد): verify يعتمد على مجلد yodas — إن لم يوجد ننفذ استدلالاً مباشراً
    # بمرجع من بيانات الاختبار المعزولة أو من أي بيانات متاحة
    ref = None
    for cand_dir in (TEST_DIR, DATA_DIR / "yodas", DATA_DIR / "youssef", DATA_DIR / "safaa"):
        if cand_dir.exists():
            cands = sorted(cand_dir.glob("*.ogg"))[:1] or sorted(cand_dir.glob("*.wav"))[:1]
            if cands:
                ref = cands[0]
                break
    if ref is not None:
        print(f"\n[احتياط] استدلال مباشر بمرجع: {ref.name}")
        outdir = RUN_DIR / "verify_out"
        outdir.mkdir(parents=True, exist_ok=True)
        sh([VENV_PY, "inference.py",
            "--checkpoint", FT,
            "--config", CFG_TINY,
            "--source", SEEDVC / "examples" / "source" / "source_s1.wav",
            "--target", ref,
            "--output", outdir,
            "--diffusion-steps", "16", "--fp16", "true"],
           cwd=SEEDVC, check=False)
        outs = sorted(outdir.glob("vc_*.wav"))

if outs:
    print(f"\nناتج التحويل: {outs[0].name}  ({outs[0].stat().st_size/1e6:.1f} MB)")
    print("استمع للناتج (المصدر إنجليزي يحمل الطابع الصوتي العراقي المرجعي):")
    display(Audio(filename=str(outs[0])))
    for extra in outs[1:3]:
        print(f"ناتج إضافي: {extra.name}")
        display(Audio(filename=str(extra)))
    print(f"\n[نجاح] النقطة المضبوطة تعمل على جهازك — انتقل إلى التقييم الكمي (الخلية 18).")
else:
    print("لم يُنتج ملف صوتي — راجع السجل أعلاه. (غير قاتل: الأوزان ستصدَّر في الخلية 20 على أي حال)")
