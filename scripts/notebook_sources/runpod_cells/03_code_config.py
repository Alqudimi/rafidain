# ============================================================ #
#  الخلية ١ — الإعدادات المركزية والأدوات المساعدة              #
#  كل مقابض التحكم في مكان واحد، وتُحفظ على القرص الدائم        #
#  (إعادة تشغيل النواة لا تفقدها — وتُعاد عند الاستئناف)         #
# ============================================================ #
import os, sys, json, subprocess, shutil, time, re
from pathlib import Path

NB_VERSION = "2.1"

# ---------- 1) جذر العمل: على تخزين دائم إن وُجد ----------
# على RunPod: القرص الشبكي (Network Volume) يُوصل تلقائياً في /workspace وهو دائم.
def _pick_root():
    # IRAQI_DATA_ROOT = مسار جذر العمل كاملاً مباشرة (نفس دلالته في سكربتات المستودع)
    override = os.environ.get("IRAQI_DATA_ROOT", "").strip()
    if override:
        return Path(override)
    for cand in ("/workspace", "/root"):
        p = Path(cand)
        try:
            p.mkdir(parents=True, exist_ok=True)
            (p / ".write_test").write_text("ok"); (p / ".write_test").unlink()
            return p / "iraqi_seedvc"
        except Exception:
            continue
    return Path.home() / "iraqi_seedvc"

ROOT = _pick_root()   # /workspace/iraqi_seedvc (دائم) أو /root/iraqi_seedvc (مؤقت)
ROOT.mkdir(parents=True, exist_ok=True)
CFG_PATH  = ROOT / "nb_config.json"

# ---------- 2) المقابض الافتراضية ----------
DEFAULTS = dict(
    # البيانات
    YODAS_HOURS    = 200,    # ساعات YODAS العراقية (0 = تعطيل، "all" = الكل 339 ساعة، أو أي رقم)
    SKIP_SMALL     = False,  # تخطي المجموعات النظيفة الخمس (youssef/shiekh/safaa/lahgtna/kharrufa)
    # التدريب
    RUN_NAME       = "iraqi_v1",  # اسم التشغيل (مجلد النقاط والحزمة)
    STEPS          = 3000,   # عدد خطوات الضبط الدقيق (500 لمتحدث واحد، 1500-4000 للهجة عموماً)
    BATCH          = "auto", # حجم الدفعة: "auto" = حسب ذاكرة GPU، أو رقم 1/2/4/6/8
    SAVE_EVERY     = 250,    # حفظ نقطة تفتيش كل N خطوة (يُحتفظ بآخر نقطتين تلقائياً)
    TRAIN_WORKERS  = 2,      # عمال تحميل بيانات التدريب
    # التنزيل والتجهيز
    FETCH_WORKERS  = 16,     # تنزيلات YODAS المتوازية (ارفعها إلى 32 إذا كانت الشبكة سريعة)
    PREP_WORKERS   = 0,      # عمال التقطيع والترميز (0 = تلقائي: حتى 16 نواة)
    HF_ENDPOINT    = "",     # مرآة HuggingFace إن لزم (مثال: https://hf-mirror.com) — اتركها فارغة عادة
    # التقييم (جديد في الإصدار ٢)
    HOLDOUT_COUNT  = 24,     # عدد مقاطع الاختبار المعزولة عن التدريب (تقييم أمين)
    EVAL_PAIRS     = 5,      # أزواج A/B في التقييم الذهبي
    EVAL_STEPS     = 25,     # خطوات التدفق أثناء التقييم (متساوية للنموذجين = مقارنة عادلة)
    # التخزين الدائم (HuggingFace Hub)
    HF_TOKEN       = "",     # رمز من hf.co/settings/tokens بصلاحية Write — للرفع الدائم (الخلية 22)
    HF_REPO_ID     = "",     # فارغ = <اسم-حسابك>/rafidain-seedvc-iraqi تلقائياً
    HF_PRIVATE     = True,   # مستودع خاص لا يراه أحد غيرك
    # تنظيف
    CLEAN_RAW_AFTER = False, # حذف المصادر الخام بعد تجهيز بيانات التدريب (يوفر مساحة — الخلية 24)
)

# ---------- 3) إعادة تحميل الإعدادات المحفوظة (استئناف بعد إعادة تشغيل النواة) ----------
CFG = dict(DEFAULTS)
FORCE_RESET = False   # اجعلها True مؤقتاً لإعادة كل الإعدادات إلى الافتراضية
if CFG_PATH.exists() and not FORCE_RESET:
    try:
        saved = json.loads(CFG_PATH.read_text(encoding="utf-8"))
        CFG.update({k: saved[k] for k in DEFAULTS if k in saved})
        print(f"[إعدادات] عيد تحميلها من {CFG_PATH} (عدّل ثم أعد التشغيل، أو FORCE_RESET=True للبدء من جديد)")
    except Exception as e:
        print(f"[إعدادات] تعذر قراءة المحفوظة ({e}) — نستخدم الافتراضية")
# ← عدّل أي قيمة هنا ثم أعد تشغيل الخلية، مثال:
#    CFG["YODAS_HOURS"] = 50
#    CFG["STEPS"] = 1000
CFG["NB_VERSION"] = NB_VERSION
CFG_PATH.write_text(json.dumps(CFG, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- 4) المسارات الثابتة (كلها تحت الجذر الدائم) ----------
SEEDVC   = ROOT / "seed-vc"                    # مستودع Seed-VC (المدرب الأصلي)
RAFIDAIN = ROOT / "rafidain"                   # مستودع رافِدَين (الخدمة + خط البيانات)
PIPE     = ROOT / "iraqi_pipeline.py"          # خط بيانات العراق — يُستخرج في الخلية 6
DATA_DIR = ROOT / "data" / "iraqi_finetune"    # بيانات التدريب المجهزة (Opus 24kHz)
TEST_DIR = ROOT / "data" / "iraqi_test"        # مقاطع اختبار معزولة لا يدخلها التدريب
RUN_DIR  = SEEDVC / "runs" / CFG["RUN_NAME"]   # نقاط التدريب والسجلات
EXPORTS  = ROOT / "exports"                    # حزم النموذج النهائية
VENV_PY  = ROOT / "venv" / "bin" / "python"    # بايثون البيئة المعزولة (الخلية 4)
for p in (ROOT, EXPORTS, RUN_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------- 5) بيئة الأوامر وأداة تشغيل موحدة ----------
ENV = dict(os.environ)
ENV["PYTHONUNBUFFERED"] = "1"
ENV["HF_HUB_DISABLE_XET"] = "1"           # تنزيل HTTP قابل للاستئناف (نفس خيار سكربتات المستودع)
ENV["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
if CFG["HF_ENDPOINT"]:
    ENV["HF_ENDPOINT"] = CFG["HF_ENDPOINT"]

def sh(cmd, cwd=None, env=None, check=True, echo=True):
    # تشغيل أمر وطباعة مخرجاته لحظياً داخل الخلية — يعيد رمز الخروج
    cmd = [str(c) for c in cmd]
    if echo:
        print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None,
                            env=env or ENV, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    rc = proc.wait()
    if check and rc != 0:
        raise RuntimeError(f"فشل الأمر (رمز {rc}): {' '.join(cmd)}")
    return rc

# ---------- 6) حارس بيانات الاختبار: يُستدعى قبل التدريب لمنع تسريبها ----------
def reapply_holdout(verbose=True):
    """أعد فصل مقاطع الاختبار إن عادت لمجلد التدريب (بعد إعادة تشغيل prep مثلاً)."""
    hl = ROOT / "data" / "holdout_list.json"
    if not hl.exists():
        return 0
    moved = 0
    for rel in json.loads(hl.read_text()):
        src = DATA_DIR / rel
        if src.exists():
            TEST_DIR.mkdir(parents=True, exist_ok=True)
            src.rename(TEST_DIR / src.name)
            moved += 1
    if moved and verbose:
        print(f"[اختبار] أُعيد عزل {moved} مقطعاً إلى {TEST_DIR} (منع تسريب للتدريب)")
    return moved

# ---------- 7) ملخص ----------
print("=" * 64)
print(f"الجذر الدائم للعمل : {ROOT}")
print("=" * 64)
print(f"  • ساعات YODAS        : {CFG['YODAS_HOURS']}")
print(f"  • التدريب            : {CFG['STEPS']} خطوة، حفظ كل {CFG['SAVE_EVERY']}, دفعة {CFG['BATCH']}")
print(f"  • التقييم            : {CFG['EVAL_PAIRS']} أزواج A/B على {CFG['HOLDOUT_COUNT']} مقطعاً معزولاً")
print(f"  • اسم التشغيل        : {CFG['RUN_NAME']}")
print(f"  • رمز HuggingFace    : {'مضبوط (سيُرفع للأبد في الخلية 22)' if CFG['HF_TOKEN'] else 'غير مضبوط (اختياري — الرفع الدائم)'}")
print(f"\n[المسارات]  التدريب : {DATA_DIR}")
print(f"            الاختبار: {TEST_DIR}")
print(f"            النقاط  : {RUN_DIR}")
print(f"            التصدير : {EXPORTS}")
print(f"\nملاحظة: الرمز السري (إن وضعته) يُحفظ في {CFG_PATH} على قرصك الدائم فقط.")
