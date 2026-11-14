# ============================================================ #
#  الخلية ١٤ — التدريب المباشر (استيراد train.py كوحدة بايثون)   #
#  استدعاء مباشر لـ Trainer.train() بلا subprocess            #
#  التقدم يظهر مباشرة في الخلية ويُحفظ في train_console.log     #
# ============================================================ #
import os, sys, time, json, io, contextlib
from pathlib import Path
import torch

LOG = RUN_DIR / "train_console.log"
FT = RUN_DIR / "ft_model.pth"
CFG_TINY = SEEDVC / "configs" / "presets" / "config_dit_mel_seed_uvit_xlsr_tiny.yml"

# ---- حراس قبل الانطلاق ----
assert torch.cuda.is_available(), "لا يوجد GPU متاح للنواة! التدريب يتطلب حاوية GPU على RunPod."
assert (SEEDVC / "train.py").exists(), "مستودع seed-vc غير موجود — شغّل الخلية 5"
assert DATA_DIR.exists() and sum(1 for _ in DATA_DIR.rglob("*.ogg")) >= 50, \
    "بيانات تدريب غير كافية — شغّل خلايا البيانات (7-10)"
_n_test = len(list(TEST_DIR.glob("*.ogg"))) if TEST_DIR.exists() else 0
if _n_test == 0:
    print("[تنبيه] لا توجد بيانات اختبار معزولة — خلية التقييم A/B (18) ستعمل على عينة من بيانات التدريب.")

# ---- منع تسريب بيانات الاختبار ----
reapply_holdout()

BATCH = int(CFG.get("_BATCH") or 2)
MAXS = int(CFG["STEPS"])
SAVE_EVERY = int(CFG["SAVE_EVERY"])
WORKERS = int(CFG.get("TRAIN_WORKERS", 2))

if FT.exists():
    print(f"النموذج النهائي موجود أصلاً: {FT}")
    print("لتدريب جديد كلياً: غيّر RUN_NAME في الخلية 1 (أو احذف ft_model.pth ثم أعد هذه الخلية).")
else:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # التأكد من إدراج مسارات seed-vc في sys.path للاستيراد المباشر
    for sp in (SEEDVC, SEEDVC / "modules"):
        if str(sp) not in sys.path:
            sys.path.insert(0, str(sp))

    import train as seed_train

    class Tee(io.TextIOBase):
        def __init__(self, original, path):
            self.original = original
            self.file = open(path, "a", encoding="utf-8", buffering=1)
        def write(self, s):
            self.original.write(s)
            self.file.write(s)
            return len(s)
        def flush(self):
            self.original.flush()
            self.file.flush()

    t0 = time.time()
    print(f"[train] بدء التدريب باستدعاء seed_train.Trainer مباشرة: {MAXS} خطوة (دفعة {BATCH}، حفظ كل {SAVE_EVERY})")
    print(f"  السجل الكامل: {LOG}")
    print(f"  مجلد النقاط  : {RUN_DIR}")

    _old_cwd = os.getcwd()
    os.chdir(str(SEEDVC))
    try:
        with contextlib.redirect_stdout(Tee(sys.stdout, str(LOG))):
            trainer = seed_train.Trainer(
                config_path=str(CFG_TINY),
                pretrained_ckpt_path=None,
                data_dir=str(DATA_DIR),
                run_name=CFG["RUN_NAME"],
                batch_size=BATCH,
                num_workers=WORKERS,
                steps=MAXS,
                save_interval=SAVE_EVERY,
                max_epochs=1000,
                device="cuda:0",
            )
            trainer.train()
    finally:
        os.chdir(_old_cwd)

    mins = (time.time() - t0) / 60
    if FT.exists():
        print(f"\n[اكتمل] انتهى التدريب بنجاح في {mins:.1f} دقيقة — النموذج النهائي: {FT}")
    else:
        print(f"\n[تحذير] توقف التدريب بعد {mins:.1f} دقيقة — راجع السجل: {LOG}")
