# ============================================================ #
#  الخلية ١٤ — إطلاق التدريب (عملية خلفية مستقلة عن النواة)    #
#  لا تتأثر بإغلاق المتصفح أو انقطاعك أو إعادة تشغيل النواة    #
#  التقدم يُكتب في: runs/<run>/train_console.log               #
# ============================================================ #
import subprocess, os, time, json
from pathlib import Path

LOG = RUN_DIR / "train_console.log"
PIDF = RUN_DIR / "train.pid"
FT = RUN_DIR / "ft_model.pth"
CFG_TINY = SEEDVC / "configs" / "presets" / "config_dit_mel_seed_uvit_xlsr_tiny.yml"

def train_alive():
    if not PIDF.exists():
        return False
    try:
        os.kill(int(PIDF.read_text().strip()), 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False

# ---- حراس قبل الانطلاق (جديد في الإصدار ٢) ----
import shutil as _sh
_gpus = []
try:
    _out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=30).stdout.strip()
    _gpus = [l for l in _out.splitlines() if l.strip()]
except Exception:
    pass
assert _gpus, "لا يوجد GPU! التدريب يتطلب حاوية GPU على RunPod. (تجهيز البيانات وحده يعمل بلا GPU)"
assert (SEEDVC / "train.py").exists(), "مستودع seed-vc غير موجود — شغّل الخلية 5"
assert DATA_DIR.exists() and sum(1 for _ in DATA_DIR.rglob("*.ogg")) >= 50, \
    "بيانات تدريب غير كافية — شغّل خلايا البيانات (7-10)"
_n_test = len(list(TEST_DIR.glob("*.ogg"))) if TEST_DIR.exists() else 0
if _n_test == 0:
    print("[تنبيه] لا توجد بيانات اختبار معزولة — خلية التقييم A/B (18) ستعمل على عينة من بيانات التدريب.")
    print("         يُنصح بتشغيل الخلية 12 أولاً لتقييم أمين. (يمكنك المتابعة الآن إن أردت)")

# ---- منع تسريب بيانات الاختبار (إن عادت بعد إعادة تجهيز) ----
reapply_holdout()

BATCH = int(CFG.get("_BATCH") or 2)
MAXS = int(CFG["STEPS"])

if FT.exists():
    print(f"النموذج النهائي موجود أصلاً: {FT}")
    print("لتدريب جديد كلياً: غيّر RUN_NAME في الخلية 1 (أو احذف ft_model.pth ثم أعد هذه الخلية).")
elif train_alive():
    print(f"التدريب يعمل الآن (PID {PIDF.read_text().strip()}) — راقبه بالخلية 15.")
else:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if PIDF.exists():
        PIDF.unlink()   # بقايا عملية ميتة
    cmd = [str(VENV_PY), "-u", "train.py",
           "--config", str(CFG_TINY),
           "--dataset-dir", str(DATA_DIR),
           "--run-name", CFG["RUN_NAME"],
           "--batch-size", str(BATCH),
           "--max-steps", str(MAXS),
           "--max-epochs", "1000",
           "--save-every", str(int(CFG["SAVE_EVERY"])),
           "--num-workers", str(int(CFG["TRAIN_WORKERS"])),
           "--gpu", "0"]
    print("أمر التدريب:")
    print("  cd " + str(SEEDVC) + " && " + " ".join(cmd) + "\n")
    log_fh = open(LOG, "a")
    proc = subprocess.Popen(cmd, cwd=str(SEEDVC), env=ENV,
                            stdout=log_fh, stderr=subprocess.STDOUT,
                            start_new_session=True)   # يستقل عن النواة نهائياً
    PIDF.write_text(str(proc.pid))
    time.sleep(3)
    if proc.poll() not in (None, 0):
        tail = LOG.read_text(errors="replace")[-3000:]
        raise RuntimeError("ماتت عملية التدريب فوراً! آخر السجل:\n" + tail)
    print(f"[انطلق] التدريب يعمل الآن في الخلفية — PID {proc.pid}")
    print(f"  السجل الكامل : {LOG}")
    print(f"  النقاط        : {RUN_DIR}")
    print(f"  الهدف         : {MAXS} خطوة (دفعة {BATCH}، حفظ كل {int(CFG['SAVE_EVERY'])})")
    print(f"  بيانات الاختبار المعزولة: {_n_test} مقطعاً (لن يدخلها التدريب)")
    print("\nالخطوة التالية: شغّل خلية «مراقبة التدريب» (15) الآن، وأعد تشغيلها كل فترة.")
    print("ويمكنك إغلاق المتصفح مطمئناً — التدريب مستقل عنه تماماً.")
