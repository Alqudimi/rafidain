# -*- coding: utf-8 -*-
"""Part A: title/prep/map markdown + code cells 1-7 (Drive, config, env, deps, venv, repos, pipeline)."""

CELLS_A = []

# ============================================================ MD 1
CELLS_A.append(("markdown", r'''<div dir="rtl">

# رافِدَين — تدريب Seed-VC على اللهجة العراقية على Google Colab

دفتر Colab واحد متكامل ينجز **الرحلة كاملة** — من جلسة GPU فارغة إلى نموذج
`seed-uvit-tat-xlsr-tiny` مضبوط بدقة على الكلام العراقي، محفوظاً بشكل **دائم** في
Google Drive و HuggingFace Hub:

| المرحلة | ماذا يحدث |
|---|---|
| 1. البيئة | توصيل Drive، فحص GPU/القرص، تثبيت ffmpeg و uv و PyTorch 2.4.0 (cu121) في بيئة معزولة |
| 2. المستودعات | استنساخ Seed-VC ورافِدَين، واستخراج خط بيانات العراق المدمج (49,194 مقطعاً / 339 ساعة) |
| 3. البيانات | تنزيل 5 مجموعات عراقية نظيفة (منها ≈50 ساعة إضافية من Lahgtna v2 + استوديو الخرّوفا) + جلب مُجزَّأ (ranged HTTP) لساعات YODAS العراقية، ثم تقطيع وترميز Opus 24kHz |
| 4. التدريب | ضبط دقيق يبدأ من نقطة `DiT_uvit_tat_xlsr_ema.pth` الرسمية، بتقدم حي ومنحنى خسارة، مع **نسخ احتياطي تلقائي إلى Drive أثناء التدريب** |
| 5. التحقق والتصدير | استدلال حقيقي بالنقطة الناتجة (استماع داخل الدفتر)، ثم حزمة تصدير (أوزان + إعداد + تقرير + سجلات) |
| 6. الحفظ الدائم | نسخة في **Google Drive** + أرشيف tar.gz + **رفع إلى HuggingFace Hub** (دائم فعلاً) |

> هذا الدفتر مبني مباشرة على خط التدريب الرسمي للمستودع (`seed_vc_iraqi_colab.sh`) — نفس المراحل
> ونفس الإصدارات المثبتة بالضبط — لكنه مقسّم إلى خلايا قابلة للتشغيل والمراقبة والاستئناف.
> كل خلية آمنة لإعادة التشغيل: المراحل تتخطى ما أُنجز منها (علامات `.done` وفحوص وجود)
> وتستأنف ما بقي. صُمم خصيصاً لطبيعة Colab: جلسات منقطعة، قرص محلي مؤقت، وDrive دائم بطيء.

**ضمانات الدفتر**: صحة المسار كاملة (نفس إصدارات الخط الرسمي المُجرَّب)، استئناف بعد أي انقطاع،
تحقق آلي قبل التصدير، وحفظ ثلاثي الطبقات لا يضيع منه شيء. جودة المخرج النهائية دالةٌ على حجم
البيانات (200 ساعة افتراضياً من أصل 339)، وعدد الخطوات (3000)، وذاكرة GPU — وكلها مقابض قابلة
للرفع في الخلية 2 إن أردت المزيد.

**المصدر**: [github.com/Alqudimi/rafidain](https://github.com/Alqudimi/rafidain) ·
النموذج: [Plachta/Seed-VC](https://huggingface.co/Plachta/Seed-VC)‏ (`seed-uvit-tat-xlsr-tiny`)

</div>'''))

# ============================================================ MD 2
CELLS_A.append(("markdown", r'''<div dir="rtl">

## ١) تجهيز Colab (قبل تشغيل الخلايا)

### أ. اختيار الـ GPU
من القائمة: **Runtime ← Change runtime type ← Hardware accelerator: GPU** ثم Save.

| GPU | كيف تحصل عليه | الذاكرة | حجم الدفعة المقترح | ملاحظات |
|---|---|---|---|---|
| **T4** | المجاني الافتراضي | 16GB | 2 | يعمل كاملاً — 3000 خطوة في نحو 2–4 ساعات |
| **L4** | Colab Pro | 24GB | 4 | أسرع بنحو الضعف من T4 |
| **A100 40GB** | Colab Pro / Pro+ | 40GB | 6 | فائض عن حاجة نموذج 25M باراميتر |

الدفتر يضبط حجم الدفعة تلقائياً حسب الذاكرة المكتشفة (الخلية 3) — لا تحتاج تعديل شيء.

### ب. حدود جلسة Colab — و كيف يتعامل معها الدفتر (بصراحة)
- الجلسة المجانية تعيش **حتى ~12 ساعة**، وتُقطع بعد **~90 دقيقة خمول** (لا نشاط في النواة).
- **الحلول المطبقة هنا**: خلية «انتظار الاكتمال» (16) تبقي النواة نشطة أثناء التدريب؛
  **مراقب النسخ** (الخلية 14) ينسخ كل نقطة تفتيش جديدة إلى Drive فور ظهورها (كل 250 خطوة) —
  أي أن أقصى خسارة عند أي انقطاع = آخر 250 خطوة فقط.
- بعد أي انقطاع: **Runtime ← Run all** — الخلايا تُعيد بناء البيئة، والخلية 13 تستعيد النقاط
  من Drive، والتدريب يُكمل من آخر نقطة حفظ تلقائياً. البيانات الخام تُعاد جلبها فقط إذا فُقد
  القرص المحلي (كل التنزيلات قابلة للاستئناف).

### ج. Google Drive — طبقة التخزين الدائم
- تحتاج **~2GB** من حصتك (آخر نقطتين ≈ 0.4GB لكل نقطة + النموذج النهائي + السجلات + حزمة
  التصدير). حصة Drive المجانية (15GB) تكفي تماماً.
- بيانات التدريب الثقيلة (~30GB لـ 200 ساعة) تبقى على **القرص المحلي السريع** `/content` —
  عمداً: Drive بطيء جداً لآلاف الملفات الصغيرة، والبيانات قابلة لإعادة الجلب في أي وقت.

### د. قرص Colab المحلي
| الإعداد | تنزيل YODAS الخام | الإجمالي التقريبي (بيانات + بيئة + نقاط) |
|---|---|---|
| 50 ساعة | ≈ 3.1GB | ≈ 12GB |
| **200 ساعة (الافتراضي)** | **≈ 12.7GB** | **≈ 30GB** |
| 334 ساعة (الكتالوج كله) | ≈ 22.4GB | ≈ 42GB |

القرص المحلي في Colab عادة 100GB+ — كل الإعدادات أعلاه تدخل بارتياح.

### هـ. فتح هذا الدفتر
نزّل ملف `rafidain_colab_training.ipynb` وارفعه إلى [colab.research.google.com](https://colab.research.google.com)
(أو ضعه في Drive وافتحه بـ *Open with ← Colab*)، فعّل GPU كما في (أ)، ثم شغّل الخلايا من الأعلى للأسفل.

</div>'''))

# ============================================================ MD 3
CELLS_A.append(("markdown", r'''<div dir="rtl">

## ٢) خريطة الدفتر — شغّل بالترتيب من الأعلى

| # | الخلية | الوصف | المدة التقريبية |
|---|---|---|---|
| 1 | توصيل Google Drive | طبقة التخزين الدائم (تخطَّها إن أردت العمل بلا Drive) | ثوانٍ |
| 2 | الإعدادات المركزية | كل مقابض التحكم + الأدوات — تُحفظ على Drive | ثوانٍ |
| 3 | فحص البيئة | GPU/قرص/أدوات + تحديد حجم الدفعة تلقائياً | ثوانٍ |
| 4 | تبعيات النظام | ffmpeg + git + uv | أقل من دقيقة |
| 5 | بيئة Python | venv مستقل 3.11 + PyTorch 2.4.0 cu121 + الاعتماديات | 3–8 دقائق |
| 6 | المستودعات | استنساخ Seed-VC + رافِدَين | أقل من دقيقة |
| 7 | خط البيانات | استخراج خط بيانات العراق المدمج + إحصاءات الكتالوج | ثوانٍ |
| 8 | المجموعات النظيفة | youssef + shiekh + safaa (نحو 3GB) | 2–10 دقائق |
| 9 | بيانات YODAS | جلب مُجزَّأ بحسب الساعات المطلوبة (12.7GB لـ 200 ساعة) | 15–60 دقيقة |
| 10 | بياناتك الخاصة | (اختياري) مجلد لملفاتك الصوتية يُدمج في التدريب | حسبك |
| 11 | التقطيع والترميز | تقطيع 1–29.5 ثانية وترميز Opus 24kHz أحادي | 5–20 دقيقة |
| 12 | نقاط النموذج | تنزيل DiT/XLSR/HiFT/CAMPPlus مسبقاً (نحو 2.2GB) | 2–10 دقائق |
| 13 | استعادة من Drive | (بعد أي انقطاع) إعادة نقاط التدريب من Drive — ثم يستأنف التدريب | ثوانٍ |
| 14 | إطلاق التدريب | عملية خلفية + **مراقب نسخ تلقائي إلى Drive** | فوري |
| 15 | مراقبة التدريب | تقدم + ETA + منحنى الخسارة — أعد تشغيلها متى شئت | حسب الرغبة |
| 16 | انتظار الاكتمال | مراقبة تلقائية تُبقي الجلسة حية حتى النهاية | مدة التدريب |
| 17 | التحقق | استدلال حقيقي + استماع داخل الدفتر | 1–3 دقائق |
| 18 | التصدير | حزمة أوزان + إعداد + تقرير → ونسخها إلى Drive | ثوانٍ |
| 19 | أرشيف tar.gz | ملف واحد في Drive جاهز للتنزيل | ثوانٍ |
| 20 | HuggingFace Hub | **التخزين السحابي الدائم** — النموذج في حسابك للأبد | 1–5 دقائق |
| 21 | تنظيف المساحة | (اختيارية) حذف المصادر الخام القابلة لإعادة الجلب | ثوانٍ |
| 22 | النشر والاستخدام | أوامر جاهزة: استدلال فوري + خدمة رافِدَين على نفس الجلسة | اختيارية |

**قواعد ذهبية**
- كل الخلايا **آمنة لإعادة التشغيل** — لا شيء يُعاد تنفيذه إن كان منجزاً (علامات `.done` وفحوص وجود).
- إذا انقطعت الجلسة كلها: افتح الدفتر وشغّل **Run all** — البيئة تُبنى، والخلية 13 تستعيد نقاط
  التدريب من Drive، والتدريب يُكمل من آخر نقطة حفظ.
- التدريب يعمل كعملية خلفية مستقلة عن النواة؛ راقبه بالخلية 15، ودعه يعمل مع خلية 16 المبقية للجلسة.
- **لا تعتمد على القرص المحلي `/content` للديمومة** — كل ما يهم يُنسخ تلقائياً إلى Drive (الخلية 14).

</div>'''))

# ============================================================ CODE 1 — Drive mount
CELLS_A.append(("code", r'''# ============================================================ #
#  الخلية ١ — توصيل Google Drive (طبقة التخزين الدائم)         #
#  هنا تُحفظ: الإعدادات + نقاط التدريب + حزم النموذج النهائية   #
#  (يمكن تخطيها — الدفتر يعمل بلا Drive لكن لا شيء ينجو من      #
#   إعادة تعيين الجلسة إلا مرفوعاً إلى HuggingFace لاحقاً)      #
# ============================================================ #
from pathlib import Path
import shutil

DRIVE_MNT = Path("/content/drive")
DRIVE_OK, DRIVE_ROOT = False, None

try:
    from google.colab import drive          # متاح على Colab فقط
    drive.mount(str(DRIVE_MNT))
    DRIVE_OK = (DRIVE_MNT / "MyDrive").is_dir()
except ModuleNotFoundError:
    print("[Drive] لست على Colab — تخطّي توصيل Drive (العمل محلي بالكامل).")
except Exception as e:
    print(f"[Drive] تعذر التوصيل: {e} — تابع بلا Drive أو أعد هذه الخلية.")

if DRIVE_OK:
    DRIVE_ROOT = DRIVE_MNT / "MyDrive" / "rafidain_seedvc"
    for sub in ("runs", "exports"):
        (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)
    try:
        du = shutil.disk_usage(DRIVE_ROOT)
        print(f"[Drive] موصول — التخزين الدائم: {DRIVE_ROOT}")
        print(f"[Drive] الحصة: حر {du.free/1e9:.1f}GB من {du.total/1e9:.1f}GB (النسخ الاحتياطية تحتاج ~2GB)")
    except Exception:
        print(f"[Drive] موصول — التخزين الدائم: {DRIVE_ROOT}")
    print("[Drive] ما يُحفظ هنا: nb_config.json + runs/<اسم-التشغيل>/ + exports/<اسم-التشغيل>/")
else:
    print("""
[تحذير] العمل بلا Drive: كل ما يبقى على /content يضيع عند انقطاع الجلسة.
         يبقى خيارك الدائم الوحيد حينها الرفع إلى HuggingFace Hub (الخلية 20).
         لتوصيل Drive لاحقاً: شغّل هذه الخلية ثم أعد تشغيل الخلية 2.""")'''))

# ============================================================ CODE 2 — central config
CELLS_A.append(("code", r'''# ============================================================ #
#  الخلية ٢ — الإعدادات المركزية والأدوات المساعدة              #
#  كل مقابض التحكم في مكان واحد، وتُحفظ على Drive إن وُجد       #
#  (إعادة تشغيل النواة/الجلسة لا تفقدها — وتُعاد عند الاستئناف) #
# ============================================================ #
import os, sys, json, subprocess, shutil, time, re
from pathlib import Path

# ---------- 1) حالة Drive من الخلية 1 (آمنة بعد إعادة تشغيل النواة) ----------
DRIVE_OK   = globals().get("DRIVE_OK", False)
DRIVE_ROOT = globals().get("DRIVE_ROOT", None)

# ---------- 2) جذر العمل: القرص المحلي السريع دائماً (Drive بطيء للبيانات!) ----------
# على Colab: /content هو القرص المحلي. IRAQI_DATA_ROOT = مسار مخصص (نفس دلالته في سكربتات المستودع).
ROOT = Path(os.environ.get("IRAQI_DATA_ROOT", "").strip() or "/content/iraqi_seedvc")
ROOT.mkdir(parents=True, exist_ok=True)

# الإعدادات تُحفظ على Drive إن كان موصولاً — فتنجو من إعادة تعيين الجلسة
CFG_DIR  = DRIVE_ROOT if (DRIVE_OK and DRIVE_ROOT) else ROOT
CFG_PATH = CFG_DIR / "nb_config.json"

# ---------- 3) المقابض الافتراضية ----------
DEFAULTS = dict(
    # البيانات
    YODAS_HOURS    = 200,    # ساعات YODAS العراقية (0 = تعطيل، "all" = الكل 334 ساعة، أو أي رقم)
    SKIP_SMALL     = False,  # تخطي المجموعات النظيفة الخمس (youssef/shiekh/safaa/lahgtna/kharrufa)
    # التدريب
    RUN_NAME       = "iraqi_v1",  # اسم التشغيل (مجلد النقاط والحزمة)
    STEPS          = 3000,   # عدد خطوات الضبط الدقيق (500 لمتحدث واحد، 1500-4000 للهجة عموماً)
    BATCH          = "auto", # حجم الدفعة: "auto" = حسب ذاكرة GPU، أو رقم 1/2/4/6/8
    SAVE_EVERY     = 250,    # حفظ نقطة تفتيش كل N خطوة (تُنسخ تلقائياً إلى Drive)
    TRAIN_WORKERS  = 2,      # عمال تحميل بيانات التدريب
    # التنزيل
    FETCH_WORKERS  = 16,     # تنزيلات YODAS المتوازية (ارفعها إلى 32 إذا كانت الشبكة سريعة)
    PREP_WORKERS   = 0,      # عمال التقطيع والترميز (0 = تلقائي: حتى 16 نواة)
    HF_ENDPOINT    = "",     # مرآة HuggingFace إن لزم (مثال: https://hf-mirror.com) — اتركها فارغة عادة
    # النسخ الاحتياطي إلى Google Drive
    DRIVE_BACKUP   = True,   # مراقب النسخ التلقائي أثناء التدريب (الخلية 14)
    DRIVE_CKPT_KEEP = 2,     # كم نقطة تفتيش نُبقي على Drive (نقطة ≈ 0.4GB)
    # التخزين الدائم (HuggingFace Hub)
    HF_TOKEN       = "",     # رمز من hf.co/settings/tokens بصلاحية Write — للرفع الدائم (الخلية 20)
    HF_REPO_ID     = "",     # فارغ = <اسم-حسابك>/rafidain-seedvc-iraqi تلقائياً
    HF_PRIVATE     = True,   # مستودع خاص لا يراه أحد غيرك
    # تنظيف
    CLEAN_RAW_AFTER = False, # حذف المصادر الخام بعد تجهيز بيانات التدريب (يوفر مساحة — الخلية 21)
)

# ---------- 4) إعادة تحميل الإعدادات المحفوظة (استئناف بعد أي انقطاع) ----------
CFG = dict(DEFAULTS)
FORCE_RESET = False   # اجعلها True مؤقتاً لإعادة كل الإعدادات إلى الافتراضية
if CFG_PATH.exists() and not FORCE_RESET:
    try:
        saved = json.loads(CFG_PATH.read_text(encoding="utf-8"))
        CFG.update({k: saved[k] for k in DEFAULTS if k in saved})
        print(f"[إعدادات] أُعيد تحميلها من {CFG_PATH} (عدّل ثم أعد التشغيل، أو FORCE_RESET=True للبدء من جديد)")
    except Exception as e:
        print(f"[إعدادات] تعذر قراءة المحفوظة ({e}) — نستخدم الافتراضية")
# ← عدّل أي قيمة هنا ثم أعد تشغيل الخلية، مثال:
#    CFG["YODAS_HOURS"] = 50
#    CFG["STEPS"] = 1000
CFG_PATH.write_text(json.dumps(CFG, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- 5) المسارات الثابتة ----------
SEEDVC   = ROOT / "seed-vc"                    # مستودع Seed-VC (المدرب الأصلي)
RAFIDAIN = ROOT / "rafidain"                   # مستودع رافِدَين (الخدمة + خط البيانات)
PIPE     = ROOT / "iraqi_pipeline.py"          # خط بيانات العراق — يُستخرج في الخلية 7
DATA_DIR = ROOT / "data" / "iraqi_finetune"    # بيانات التدريب المجهزة (Opus 24kHz)
RUN_DIR  = SEEDVC / "runs" / CFG["RUN_NAME"]   # نقاط التدريب والسجلات (محلي سريع)
EXPORTS  = ROOT / "exports"                    # حزم النموذج النهائية (محلي)
VENV_PY  = ROOT / "venv" / "bin" / "python"    # بايثون البيئة المعزولة (الخلية 5)

# مراياها على Drive (تُنشأ عند الحاجة — لا الآن إن لم يكن موصولاً)
def drive_run_dir():
    return (DRIVE_ROOT / "runs" / CFG["RUN_NAME"]) if (DRIVE_OK and DRIVE_ROOT) else None
def drive_export_base():
    return (DRIVE_ROOT / "exports") if (DRIVE_OK and DRIVE_ROOT) else None

for p in (ROOT, EXPORTS, RUN_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------- 6) بيئة الأوامر وأداة تشغيل موحدة ----------
ENV = dict(os.environ)
ENV["PYTHONUNBUFFERED"] = "1"
ENV["HF_HUB_DISABLE_XET"] = "1"           # تنزيل HTTP قابل للاستئناف — Xet يقيّد IPs Colab (خيار الخط الرسمي)
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

# ---------- 7) ملخص ----------
print("=" * 64)
print(f"جذر العمل (محلي سريع) : {ROOT}")
print(f"التخزين الدائم (Drive) : {DRIVE_ROOT if DRIVE_OK else 'غير موصول — اعتمد على الخلية 20 لاحقاً'}")
print("=" * 64)
print(f"  • ساعات YODAS        : {CFG['YODAS_HOURS']}")
print(f"  • التدريب            : {CFG['STEPS']} خطوة، حفظ كل {CFG['SAVE_EVERY']}, دفعة {CFG['BATCH']}")
print(f"  • اسم التشغيل        : {CFG['RUN_NAME']}")
print(f"  • نسخ Drive التلقائي : {'مفعّل (آخر %d نقاط)' % CFG['DRIVE_CKPT_KEEP'] if (DRIVE_OK and CFG['DRIVE_BACKUP']) else ('غير متاح (Drive غير موصول)' if not DRIVE_OK else 'معطّل')}")
print(f"  • رمز HuggingFace    : {'مضبوط (سيُرفع للأبد في الخلية 20)' if CFG['HF_TOKEN'] else 'غير مضبوط (اختياري — الرفع الدائم)'}")
print(f"\n[المسارات]  البيانات: {DATA_DIR}")
print(f"            التدريب : {RUN_DIR}")
print(f"            التصدير : {EXPORTS}")
print(f"\nملاحظة: الرمز السري (إن وضعته) يُحفظ في {CFG_PATH} — على Drive الخاص بك فقط.")'''))

# ============================================================ CODE 3 — env check
CELLS_A.append(("code", r'''# ============================================================ #
#  الخلية ٣ — فحص البيئة: GPU / قرص / أدوات                    #
#  (يحدد حجم الدفعة تلقائياً حسب ذاكرة GPU المكتشفة)           #
# ============================================================ #
import subprocess, os, shutil, re, json, sys
from pathlib import Path

def gpu_query():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        return [l.strip() for l in out if l.strip()]
    except Exception:
        return []

gpus = gpu_query()
if gpus:
    for i, g in enumerate(gpus):
        parts = [p.strip() for p in g.split(",")]
        name, vram_mib, driver = parts[0], float(parts[1]), parts[2]
        print(f"[GPU {i}] {name} — ذاكرة {vram_mib/1024:.1f}GB — مشغل {driver}")
    if any("T4" in g for g in gpus):
        print("        [T4] الجلسة المجانية — 3000 خطوة تستغرق نحو 2-4 ساعات. اصبر أو ترقَّ إلى L4.")
else:
    print("[GPU] لا يوجد GPU! من القائمة: Runtime <- Change runtime type <- GPU ثم أعد تشغيل")
    print("        الجلسة (Runtime <- Run all) — تجهيز البيانات يعمل بدون GPU، التدريب لا.")

def auto_batch(vram_gb):
    if vram_gb >= 70: return 8
    if vram_gb >= 40: return 6
    if vram_gb >= 20: return 4
    if vram_gb >= 15: return 2
    return 1

if gpus:
    vram_gb = float([p.strip() for p in gpus[0].split(",")][1]) / 1024
    CFG["_BATCH"] = auto_batch(vram_gb)
else:
    vram_gb = 0.0
    CFG["_BATCH"] = 2
if str(CFG["BATCH"]).lower() == "auto":
    BATCH = CFG["_BATCH"]
    print(f"[دفعة] تلقائي => BATCH = {BATCH} (حسب {vram_gb:.0f}GB ذاكرة)")
else:
    BATCH = int(CFG["BATCH"])
    print(f"[دفعة] يدوي  => BATCH = {BATCH}")
CFG["_BATCH"] = BATCH
CFG_PATH.write_text(json.dumps(CFG, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- المعالج والذاكرة ----
cpu = os.cpu_count() or 4
try:
    mem_kb = int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1))
    print(f"[نظام] {cpu} نواة معالج — ذاكرة RAM ≈ {mem_kb/1e6:.1f}GB — بايثون النواة {sys.version_info.major}.{sys.version_info.minor}")
except Exception:
    print(f"[نظام] {cpu} نواة معالج — بايثون النواة {sys.version_info.major}.{sys.version_info.minor}")

# ---- القرص المحلي (حجم البيانات الضخمة) ----
du = shutil.disk_usage(ROOT)
print(f"[قرص محلي] المتاح على {ROOT}: حر {du.free/1e9:.1f}GB من {du.total/1e9:.1f}GB")
need = {"50": 12, "200": 30, "334": 42}.get(str(CFG["YODAS_HOURS"]), 30)
if du.free / 1e9 < need + 5:
    print(f"          [تحذير] إعدادك يحتاج ~{need}GB — قلّل YODAS_HOURS أو شغّل خلية التنظيف (21)")
else:
    print(f"          [جيد] يكفي لإعدادك الحالي (~{need}GB مطلوبة تقريباً)")

# ---- Drive ----
if DRIVE_OK and DRIVE_ROOT:
    try:
        ddu = shutil.disk_usage(DRIVE_ROOT)
        print(f"[Drive] الحصة: حر {ddu.free/1e9:.1f}GB من {ddu.total/1e9:.1f}GB — النسخ الاحتياطية تحتاج ~2GB")
    except Exception:
        print("[Drive] موصول (تعذر قراءة الحصة)")

# ---- الأدوات ----
for tool in ("ffmpeg", "ffprobe", "git", "curl", "uv"):
    print(f"[أداة] {tool:<8}: {'متوفر' if shutil.which(tool) else 'غير متوفر — تثبّته الخلية 4'}")

print("\n[فحص مبدئي] أكمل إلى الخلية 4 (التبعيات).")'''))

# ============================================================ CODE 4 — system deps + uv
CELLS_A.append(("code", r'''# ============================================================ #
#  الخلية ٤ — تبعيات النظام (ffmpeg / git) + مدير الحزم uv     #
#  (Colab يعمل بصلاحيات root عادة — بلا sudo)                  #
# ============================================================ #
import shutil, os, subprocess
from pathlib import Path

def have(x):
    return shutil.which(x) is not None

missing = [p for p in ("ffmpeg", "git") if not have(p)]
if missing:
    print(f"تثبيت حزم النظام: {', '.join(missing)} ...")
    sh(["apt-get", "update", "-qq"], check=False)
    sh(["apt-get", "install", "-y", "-qq", *missing])
for tool in ("ffmpeg", "git", "curl"):
    print(f"  {tool:<8}: {'متوفر' if have(tool) else 'غير متوفر!'}")

# ---- uv: مُسرّع تثبيت بايثون (نفس مسار سكربتات المستودع الرسمية) ----
UV = shutil.which("uv")
if UV is None:
    print("\nتثبيت uv (مرة واحدة لكل جلسة) ...")
    sh(["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], check=False)
    for cand in (Path.home() / ".local" / "bin" / "uv", Path.home() / ".cargo" / "bin" / "uv"):
        if cand.exists():
            UV = str(cand)
            break
    if UV:
        os.environ["PATH"] = str(Path(UV).parent) + os.pathsep + os.environ["PATH"]
        ENV["PATH"] = os.environ["PATH"]
if UV:
    r = subprocess.run([UV, "--version"], capture_output=True, text=True)
    print(f"\nuv جاهز: {r.stdout.strip() or r.stderr.strip()}")
else:
    print("\nuv غير متوفر — سنعتمد python3 -m venv + pip كبديل في الخلية 5 (يعمل أيضاً).")'''))

# ============================================================ CODE 5 — venv + torch
CELLS_A.append(("code", r'''# ============================================================ #
#  الخلية ٥ — بيئة Python معزولة (3.11) + PyTorch 2.4.0 cu121  #
#  نفس الإصدارات المثبتة في الخط الرسمي seed_vc_iraqi_colab.sh #
#  (محلية على /content: تُعاد بناؤها بعد أي إعادة تعيين جلسة — #
#   هذا طبيعي وتستغرق 3-8 دقائق فقط بفضل تخطّي المنجز)         #
# ============================================================ #
import subprocess, shutil, sys
from pathlib import Path

UV = globals().get("UV") or shutil.which("uv")   # تحصين: يعمل حتى بعد إعادة تشغيل النواة

def venv_ok(code):
    # فحص سريع داخل البيئة المعزولة
    if not VENV_PY.exists():
        return False
    r = subprocess.run([str(VENV_PY), "-c", code], capture_output=True, text=True, env=ENV)
    return r.returncode == 0

# ---- 1) إنشاء venv إن لم يوجد ----
if not VENV_PY.exists():
    print(f"إنشاء بيئة معزولة (Python 3.11) في {ROOT/'venv'} ...")
    if UV:
        sh([UV, "venv", "--python", "3.11", str(ROOT / "venv")])
    else:
        sh([sys.executable, "-m", "venv", str(ROOT / "venv")])

# ---- 2) PyTorch 2.4.0 cu121 (نحو 2.5GB — يُتخطى إن كان مثبتاً) ----
if not venv_ok("import torch; assert torch.__version__.startswith('2.4.')"):
    print("تثبيت PyTorch 2.4.0 + torchaudio (cu121) — عدة دقائق حسب الشبكة ...")
    trio_pin = ["torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0"]
    trio_any = ["torch", "torchvision", "torchaudio"]
    idx = ["--index-url", "https://download.pytorch.org/whl/cu121"]
    if UV:
        rc = sh([UV, "pip", "install", "--python", str(VENV_PY), *trio_pin, *idx], check=False)
        if rc != 0:
            sh([UV, "pip", "install", "--python", str(VENV_PY), *trio_any, *idx])
    else:
        rc = sh([str(VENV_PY), "-m", "pip", "install", "-q", *trio_pin, *idx], check=False)
        if rc != 0:
            sh([str(VENV_PY), "-m", "pip", "install", "-q", *trio_any, *idx])
else:
    print("PyTorch 2.4.x موجود أصلاً في البيئة — تخطّي التثبيت.")

# ---- 3) بقية الاعتماديات (مثبتة بالإصدارات نفسها التي نجح بها الخط الرسمي) ----
NEED = ("import torch, torchaudio, librosa, soundfile, munch, einops,"
        " transformers, pyarrow, huggingface_hub, yaml, matplotlib")
if not venv_ok(NEED):
    print("تثبيت الاعتماديات (numpy/scipy/librosa/transformers/...) ...")
    pins = ["numpy==1.26.4", "scipy==1.13.1", "librosa==0.10.2", "soundfile==0.12.1",
            "munch==4.0.0", "einops==0.8.0", "transformers==4.46.3", "pyyaml", "tqdm",
            "requests", "pyarrow==17.0.0", "huggingface_hub", "matplotlib"]
    loose = ["numpy", "scipy", "librosa", "soundfile", "munch", "einops", "transformers",
             "pyyaml", "tqdm", "requests", "pyarrow", "huggingface_hub", "matplotlib"]
    if UV:
        rc = sh([UV, "pip", "install", "--python", str(VENV_PY), *pins], check=False)
        if rc != 0:
            sh([UV, "pip", "install", "--python", str(VENV_PY), *loose])
    else:
        rc = sh([str(VENV_PY), "-m", "pip", "install", "-q", *pins], check=False)
        if rc != 0:
            sh([str(VENV_PY), "-m", "pip", "install", "-q", *loose])
else:
    print("الاعتماديات موجودة أصلاً — تخطّي التثبيت.")

# ---- 4) تحقق نهائي ----
r = subprocess.run([str(VENV_PY), "-c",
                    "import torch, torchaudio; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"],
                   capture_output=True, text=True, env=ENV)
print((r.stdout + r.stderr).strip())
assert r.returncode == 0 and "torch 2.4" in (r.stdout + r.stderr), "فشل بناء البيئة — راجع السجل أعلاه"
print(f"\n[جاهز] البيئة المعزولة: {VENV_PY}")'''))

# ============================================================ CODE 6 — clone repos
CELLS_A.append(("code", r'''# ============================================================ #
#  الخلية ٦ — استنساخ المستودعات: Seed-VC + رافِدَين            #
#  (على القرص المحلي — يُحدثان إن وُجدا بدل إعادة التنزيل)      #
# ============================================================ #
from pathlib import Path

def git_clone(url, dst):
    if (dst / ".git").exists():
        print(f"تحديث {dst.name} ...")
        sh(["git", "-C", str(dst), "pull", "--ff-only"], check=False)
    else:
        print(f"استنساخ {dst.name} من {url} ...")
        sh(["git", "clone", "--depth", "1", url, str(dst)])

git_clone("https://github.com/Plachtaa/seed-vc.git", SEEDVC)
git_clone("https://github.com/Alqudimi/rafidain.git", RAFIDAIN)

# تحقق من الملفات الحرجة
for f in ("train.py", "inference.py", "configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml"):
    assert (SEEDVC / f).exists(), f"ملف مفقود في seed-vc: {f}"
assert (RAFIDAIN / "seed_vc_iraqi_ubuntu.sh").exists(), "سكربت خط البيانات مفقود في rafidain"
print(f"\n[جاهز] seed-vc   : {SEEDVC}")
print(f"[جاهز] rafidain  : {RAFIDAIN}")'''))

# ============================================================ CODE 7 — extract pipeline
CELLS_A.append(("code", r'''# ============================================================ #
#  الخلية ٧ — استخراج خط بيانات العراق المدمج                  #
#  (الكتالوج العراقي كله مضمّن داخل سكربت المستودع نفسه:        #
#   49,194 مقطعاً / 339 ساعة / خريطة إزاحات بايت-دقيقة          #
#   داخل أرشيفات espnet/yodas3 — نستخرجه هنا كوحدة مستقلة)     #
# ============================================================ #
import re, gzip, base64, json, ast, urllib.request
from pathlib import Path

SRC_SH = RAFIDAIN / "seed_vc_iraqi_ubuntu.sh"
if not SRC_SH.exists():   # احتياط: المستودع غير منسوخ — ننزّل السكربت مباشرة من GitHub
    print("تنزيل السكربت من GitHub مباشرة ...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/Alqudimi/rafidain/main/seed_vc_iraqi_ubuntu.sh", SRC_SH)

text = SRC_SH.read_text(encoding="utf-8")
m = re.search(r"cat > \"\$ROOT/iraqi_pipeline\.py\" <<'PYEOF'\n(.*?)\nPYEOF\n", text, re.S)
assert m, "لم يُعثر على خط البيانات المدمج داخل السكربت!"
code = m.group(1)
ast.parse(code)   # تحقق من سلامة الكود المستخرج قبل الكتابة
PIPE.write_text(code, encoding="utf-8")
print(f"[خط البيانات] استُخرج إلى: {PIPE}  ({len(code):,} بايت)")

# ---- إحصاءات الكتالوج العراقي المدمج (بدون أي تنزيل) ----
def _grab(name):
    blob = re.search(name + r' = r"""(.*?)"""', code, re.S).group(1)
    return json.loads(gzip.decompress(base64.b64decode(blob)))

clips, offs, srcrank = _grab("_IQ_CLIPS_B64"), _grab("_IQ_OFFS_B64"), _grab("_IQ_SRC_B64")
hours_total = sum(c["du"] for c in clips) / 3600
resolvable = [s for s, v in offs.items() if v.get("off")]
print(f"[كتالوج اللهجة العراقية] {len(clips):,} مقطعاً | {hours_total:.0f} ساعة | "
      f"{len(resolvable):,} تسجيل مصدر قابل للجلب | {sum(offs[s]['size'] for s in resolvable)/1e9:.1f}GB لو جُلب كله")

# ---- حجم هدفك الحالي (نفس منطق الاختيار في السكربت: الجودة أولاً) ----
t = str(CFG["YODAS_HOURS"]).strip().lower()
if t in ("0", ""):
    print(f"\n[هدفك] YODAS معطّل (YODAS_HOURS=0) — الخلية 9 ستتخطاه. الأصل المتاح: {hours_total:.0f} ساعة.")
else:
    target = None if t in ("all", "كل") else float(t)
    ranked = sorted(srcrank.items(), key=lambda kv: -kv[1][2])
    chosen, h = [], 0.0
    for sid, (tar, ch, q) in ranked:
        if sid not in offs or offs[sid].get("off") is None:
            continue
        chosen.append(sid); h += ch
        if target and h >= target:
            break
    size_gb = sum(offs[s]["size"] for s in chosen) / 1e9
    label = f"{target:.0f} ساعة" if target else "الكتالوج كله"
    print(f"[هدفك] {label}: {len(chosen):,} تسجيلاً مصدراً ≈ {size_gb:.1f}GB تنزيلاً خاماً (سيُشفَّر إلى Opus للتدريب)")
print("\n[جاهز] تابع إلى خلايا البيانات (8 وما بعدها).")'''))
