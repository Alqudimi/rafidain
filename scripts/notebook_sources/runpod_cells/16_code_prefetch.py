# ============================================================ #
#  الخلية ١٣ — تنزيل نقاط النموذج الأساسية مسبقاً (≈2.2GB)      #
#  DiT_uvit_tat_xlsr_ema + XLSR-300m + HiFT + CAMPPlus +       #
#  OpenVoice — كلها في مجلد checkpoints داخل seed-vc           #
#  (يمنع تعطل بدء التدريب لاحقاً بسبب التنزيل، وقابل للاستئناف) #
# ============================================================ #
sh([VENV_PY, PIPE, "prefetch", "--repo", SEEDVC])
import shutil
ck = SEEDVC / "checkpoints"
total = sum(f.stat().st_size for f in ck.rglob("*") if f.is_file()) / 1e9 if ck.exists() else 0
print(f"\nحجم مخزون النقاط: {total:.2f}GB في {ck}")

# حارس: تأكيد النقطة الأم التي يبدأ منها التدريب (DiT_uvit_tat_xlsr_ema.pth)
base_ema = list(ck.rglob("DiT_uvit_tat_xlsr_ema.pth")) if ck.exists() else []
assert base_ema, "نقطة الأساس DiT_uvit_tat_xlsr_ema.pth غير موجودة بعد التنزيل — أعد هذه الخلية"
print(f"[مؤكد] نقطة الأساس جاهزة: {base_ema[0]}")
print("[جاهز] كل شيء مطلوب للتدريب موجود — تابع إلى مرحلة التدريب.")
