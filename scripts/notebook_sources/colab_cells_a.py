# -*- coding: utf-8 -*-
"""Part A: title/prep/map markdown + code cells 1-6 (Drive, config, env, deps, venv, repos+imports).

v2 (2026-11-14) — "no daemons, imports only":
* No background processes, no PIDs, no wait/monitor loops, no keep-alive —
  this is what got the free Colab tier to disconnect the runtime.
* Everything is driven by importing the cloned repos' modules directly
  (iraqi_pipeline / train / inference) instead of CLI subprocess calls.
* Training runs in the foreground of its cell; a torch.save wrapper mirrors
  each checkpoint to Drive synchronously (no watcher process).
* Data comes from the prepared archive rafidain_current_sources_ready.tar.gz
  (4.89GB, iraqi_finetune/**.ogg ready for training) — the notebook downloads
  and uses it instead of re-fetching/re-preparing from the live sources.
"""

CELLS_A = []

# ============================================================ MD 1 — title
CELLS_A.append(("markdown", r'''<div dir="rtl">

# رافِدَين — تدريب Seed-VC على اللهجة العراقية على Google Colab (نسخة 2)

دفتر Colab واحد متكامل يصل من جلسة GPU فارغة إلى نموذج
`seed-uvit-tat-xlsr-tiny` مضبوط على الكلام العراقي، محفوظاً بشكل **دائم** في
Google Drive و HuggingFace Hub:

| المرحلة | ماذا يحدث |
|---|---|
| 1. البيئة | توصيل Drive، فحص GPU/القرص، تثبيت ffmpeg + حزم بايثون **داخل نواة الجلسة** (PyTorch 2.4.0 cu121) — بلا بيئة معزولة |
| 2. المستودعات | استنساخ Seed-VC + رافِدَين من GitHub، ثم **استيراد وحداتهما مباشرة** (بلا سطر أوامر) |
| 3. البيانات الجاهزة | تنزيل أرشيف **`rafidain_current_sources_ready.tar.gz`** (4.89GB) — بيانات تدريب مقطّعة ومرمّزة Opus 24kHz جاهزة، **بلا إعادة تنزيل من المصادر ولا إعادة معالجة** |
| 4. التدريب | ضبط دقيق **داخل نواة الدفتر نفسها** عبر `Trainer` المستورد (`import train`)، مع مرآة نقاط فورية إلى Drive عند كل حفظ |
| 5. التحقق والتصدير | استدلال حقيقي بالنقطة الناتجة (استماع داخل الدفتر)، ثم حزمة تصدير (أوزان + إعداد + تقرير) |
| 6. الحفظ الدائم | نسخة في **Google Drive** + أرشيف tar.gz + **رفع إلى HuggingFace Hub** (دائم فعلاً) |

> **ما الجديد في النسخة 2 ولماذا؟** النسخة السابقة كانت تُطلق التدريب كعملية خلفية
> مستقلة عن النواة، مع مراقب نسخ دائم وحلقة انتظار لساعات — وهذا النمط يُصنَّف
> «استخداماً غير مسموح» في طبقة Colab المجانية فتُقطع الجلسة فجأة
> (Runtime disconnected). هنا **كل شيء يعمل داخل الخلية نفسها**: التدريب في المقدمة
> بمخرجات حية، والنسخ إلى Drive يحدث لحظياً عند كل حفظ نقطة من داخل عملية التدريب
> نفسها — لا عمليات خفية ولا حلقات إبقاء الجلسة.

> كل خلية **آمنة لإعادة التشغيل**: ما أُنجز يُتخطى (بصمة الأرشيف + فحوص وجود)،
> والتدريب يستأنف من آخر نقطة حفظ. صُمم لطبيعة Colab: جلسات منقطعة، قرص محلي
> مؤقت، وDrive دائم.

**المصدر**: [github.com/Alqudimi/rafidain](https://github.com/Alqudimi/rafidain) ·
النموذج: [Plachta/Seed-VC](https://huggingface.co/Plachta/Seed-VC)‏ (`seed-uvit-tat-xlsr-tiny`)

</div>'''))

# ============================================================ MD 2 — setup
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

### ب. حدود جلسة Colab — وكيف يتعامل معها هذا الدفتر (بصراحة)
- الجلسة المجانية تعيش **حتى ~12 ساعة**، وتُقطع بعد خمول النواة.
- **فلسفة النسخة 2**: لا محاولة «خداع» البقاء — التدريب يعمل في مقدمة الخلية
  (النواة مشغولة بعمل حقيقي ومخرجاته تظهر أمامك مباشرة)، وكل نقطة حفظ تُنسخ
  **فوراً** إلى Drive أثناء التدريب نفسه. لو انقطعت الجلسة في أي لحظة، خسارتك
  القصوى = الخطوات منذ آخر نقطة حفظ (250 خطوة افتراضياً ≈ دقائق).
- بعد أي انقطاع: **Runtime ← Run all** — البيئة تُبنى، والأرشيف الجاهز يُستخدم
  (بصمته محفوظة فلا يُعاد تنزيله إن وُجد)، والخلية 13 تستعيد آخر نقطة من Drive،
  والتدريب يكمل من عندها.

### ج. Google Drive — طبقة التخزين الدائم
- تحتاج **~2GB** من حصتك (آخر نقطتين ≈ 0.4GB لكل نقطة + النموذج النهائي + السجلات
  + حزمة التصدير). حصة Drive المجانية (15GB) تكفي تماماً.
- بيانات التدريب الثقيلة تبقى على **القرص المحلي السريع** `/content` — عمداً:
  Drive بطيء جداً لآلاف الملفات الصغيرة، والأرشيف الجاهز يُعاد تنزيله بسرعة عند الحاجة.

### د. قرص Colab المحلي
| العنصر | الحجم |
|---|---|
| الأرشيف الجاهز (يُنزَّل ويُفك ثم يُحذف المضغوط) | 4.9GB تنزيلاً ← ≈ 5.2GB بعد الفك |
| بيئة Python + PyTorch | ≈ 5GB |
| نقاط النموذج الأساسية | ≈ 2.2GB |
| نقاط التدريب + الحزمة | ≈ 1.5GB |
| **الإجمالي التقريبي** | **≈ 14GB** |

القرص المحلي في Colab عادة 100GB+ — دخول مريح جداً (النسخة السابقة كانت تحتاج
حتى 42GB لأنها كانت تجلب المصادر الخام وتعالجها؛ الأرشيف الجاهز ألغى ذلك كله).

### هـ. فتح هذا الدفتر
نزّل ملف `rafidain_colab_training.ipynb` وارفعه إلى [colab.research.google.com](https://colab.research.google.com)
(أو ضعه في Drive وافتحه بـ *Open with ← Colab*)، فعّل GPU كما في (أ)، ثم شغّل الخلايا
من الأعلى للأسفل.

</div>'''))

# ============================================================ MD 3 — map
CELLS_A.append(("markdown", r'''<div dir="rtl">

## ٢) خريطة الدفتر — شغّل بالترتيب من الأعلى

| # | الخلية | الوصف | المدة التقريبية |
|---|---|---|---|
| 1 | توصيل Google Drive | طبقة التخزين الدائم (تخطَّها إن أردت العمل بلا Drive) | ثوانٍ |
| 2 | الإعدادات المركزية | كل المقابض + رابط الأرشيف الجاهز — تُحفظ على Drive | ثوانٍ |
| 3 | فحص البيئة | GPU/قرص/أدوات + تحديد حجم الدفعة تلقائياً | ثوانٍ |
| 4 | تبعيات النظام | ffmpeg + git (apt فقط — لا بيئة معزولة) | أقل من دقيقة |
| 5 | حزم بايثون داخل النواة | تثبيت الإصدارات المطلوبة في نواة الجلسة نفسها — **بلا بيئة منفصلة وبلا عملية خارجية للمنطق** | 2–6 دقائق |
| 6 | المستودعات + الاستيراد | استنساخ Seed-VC + رافِدَين ثم **استيراد** خط البيانات والمدرب والاستدلال | أقل من دقيقة |
| 7 | البيانات الجاهزة | تنزيل أرشيف `rafidain_current_sources_ready.tar.gz` وفكّه — **بلا أي إعادة معالجة** | 5–20 دقيقة (مرة واحدة) |
| 8 | بياناتك الخاصة | (اختياري) مجلد لملفاتك الصوتية يُقطَّع ويُدمج في التدريب | حسبك |
| 9 | إحصاءات البيانات | تقرير الساعات/الملفات لكل مصدر (من فهرس الأرشيف) | ثوانٍ |
| 10 | نقاط النموذج | تنزيل DiT/XLSR/HiFT/CAMPPlus مسبقاً (نحو 2.2GB) — عبر استيراد خط البيانات | 2–10 دقائق |
| 11 | استعادة من Drive | (بعد أي انقطاع) إعادة آخر نقطة تدريب من Drive | ثوانٍ |
| 12 | التدريب (في المقدمة) | `Trainer` المستورد يعمل **داخل نواة الدفتر مباشرة** + مرآة Drive عند كل حفظ | 2–4 ساعات (T4) |
| 13 | التحقق | استدلال حقيقي بـ `inference` المستورد **داخل النواة** + استماع في الدفتر | 1–3 دقائق |
| 14 | التصدير | حزمة أوزان + إعداد + تقرير → ونسخها إلى Drive | ثوانٍ |
| 15 | أرشيف tar.gz | ملف واحد في Drive جاهز للتنزيل | ثوانٍ |
| 16 | HuggingFace Hub | **التخزين السحابي الدائم** — النموذج في حسابك للأبد | 1–5 دقائق |
| 17 | تنظيف المساحة | (اختيارية) حذف الأرشيف المفكوك والبيئة المؤقتة | ثوانٍ |
| 18 | النشر والاستخدام | أوامر جاهزة: استدلال فوري + خدمة رافِدَين على نفس الجلسة | اختيارية |

**قواعد ذهبية**
- كل الخلايا **آمنة لإعادة التشغيل** — لا شيء يُعاد تنزيله أو تنفيذه إن كان منجزاً
  (بصمة الأرشيف + علامات `.done` + فحوص وجود).
- إذا انقطعت الجلسة: افتح الدفتر وشغّل **Run all** — الحزم تُثبَّت، والأرشيف يُفك
  فوراً من نسخة Drive إن وُجدت، والخلية 11 تستعيد نقاط التدريب، والتدريب يكمل من
  آخر نقطة حفظ.
- **لا توجد عمليات خلفية إطلاقاً** في هذا الدفتر — كل خلية تنجز عملها وتنتهي.
  هذا تحديداً هو ما يمنع قطع الجلسة «disallowed usage» في الطبقة المجانية.
- **كل منطق المشروع يعمل داخل نواة الدفتر بالاستيراد المباشر** — التدريب
  (`import train` ← `Trainer`)، والاستدلال (`import inference` ← `main()`)، وخط
  البيانات ورفع HuggingFace. لا عمليات خارجية إطلاقاً لمنطق المشروع؛ الأوامر
  الخارجية الوحيدة هي أدوات النظام نفسها (apt/git/tar/curl) لتركيب الأدوات
  واستنساخ المستودعات وفك الأرشيف — ثم يُستخدم كل ما نُزّل عبر الاستيراد.
- **لا تعتمد على القرص المحلي `/content` للديمومة** — النقاط تُنسخ إلى Drive أثناء
  التدريب لحظياً، والحزمة النهائية تُنسخ وتُؤرشف وتُرفع في الخلايا الأخيرة.

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
    for sub in ("runs", "exports", "data"):
        (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)
    try:
        du = shutil.disk_usage(DRIVE_ROOT)
        print(f"[Drive] موصول — التخزين الدائم: {DRIVE_ROOT}")
        print(f"[Drive] الحصة: حر {du.free/1e9:.1f}GB من {du.total/1e9:.1f}GB (النسخ الاحتياطية تحتاج ~2GB)")
    except Exception:
        print(f"[Drive] موصول — التخزين الدائم: {DRIVE_ROOT}")
    print("[Drive] ما يُحفظ هنا: nb_config.json + runs/<اسم-التشغيل>/ + exports/<اسم-التشغيل>/ + data/ (الأرشيف الجاهز)")
else:
    print(R"""
[تحذير] العمل بلا Drive: كل ما يبقى على /content يضيع عند انقطاع الجلسة.
         يبقى خيارك الدائم الوحيد حينها الرفع إلى HuggingFace Hub (الخلية 16).
         لتوصيل Drive لاحقاً: شغّل هذه الخلية ثم أعد تشغيل الخلية 2.""")'''))

# ============================================================ CODE 2 — central config
CELLS_A.append(("code", r'''# ============================================================ #
#  الخلية ٢ — الإعدادات المركزية والأدوات المساعدة              #
#  كل مقابض التحكم في مكان واحد، وتُحفظ على Drive إن وُجد       #
#  (إعادة تشغيل النواة/الجلسة لا تفقدها — وتُعاد عند الاستئناف) #
# ============================================================ #
import os, sys, json, shutil, time, re
from pathlib import Path

# ---------- 1) حالة Drive من الخلية 1 (آمنة بعد إعادة تشغيل النواة) ----------
DRIVE_OK   = globals().get("DRIVE_OK", False)
DRIVE_ROOT = globals().get("DRIVE_ROOT", None)

# ---------- 2) جذر العمل: القرص المحلي السريع دائماً (Drive بطيء للبيانات!) ----------
ROOT = Path(os.environ.get("IRAQI_DATA_ROOT", "").strip() or "/content/iraqi_seedvc")
ROOT.mkdir(parents=True, exist_ok=True)

# الإعدادات تُحفظ على Drive إن كان موصولاً — فتنجو من إعادة تعيين الجلسة
CFG_DIR  = DRIVE_ROOT if (DRIVE_OK and DRIVE_ROOT) else ROOT
CFG_PATH = CFG_DIR / "nb_config.json"

# ---------- 3) المقابض الافتراضية ----------
DEFAULTS = dict(
    # البيانات الجاهزة (الأرشيف المُجهَّز مسبقاً — لا تنزيل من المصادر ولا إعادة معالجة)
    READY_ARCHIVE_URL = (
        "https://drive.usercontent.google.com/download"
        "?id=1fFCx7wvabl34Z6RtYrGEYvy9zrx5U8Du&export=download&authuser=0"
        "&confirm=t&uuid=cf4cf0ad-5338-4949-af9d-c61e27fa1bf1"
        "&at=AMrWOn35Bq4LFlvhoCZfaklOunHk%3A1794664871926"
    ),   # رابط مباشر لأرشيف rafidain_current_sources_ready.tar.gz (4.89GB)
    READY_ARCHIVE_SIZE = 4890474991,  # الحجم الدقيق بالبايت — للتحقق من اكتمال التنزيل
    READY_DIR_NAME = "rafidain_current_sources_ready",  # اسم المجلد الجذري داخل الأرشيف
    # التدريب
    RUN_NAME       = "iraqi_v1",  # اسم التشغيل (مجلد النقاط والحزمة)
    STEPS          = 3000,   # عدد خطوات الضبط الدقيق (500 لمتحدث واحد، 1500-4000 للهجة عموماً)
    BATCH          = "auto", # حجم الدفعة: "auto" = حسب ذاكرة GPU، أو رقم 1/2/4/6/8
    SAVE_EVERY     = 250,    # حفظ نقطة تفتيش كل N خطوة (تُنسخ فوراً إلى Drive)
    TRAIN_WORKERS  = 2,      # عمال تحميل بيانات التدريب
    # النسخ إلى Google Drive (لحظي عند كل حفظ نقطة — بلا عملية مراقبة)
    DRIVE_BACKUP   = True,   # نسخ كل نقطة جديدة إلى Drive فور حفظها (من داخل التدريب)
    DRIVE_CKPT_KEEP = 2,     # كم نقطة تفتيش نُبقي على Drive (نقطة ≈ 0.4GB)
    # التخزين الدائم (HuggingFace Hub)
    HF_TOKEN       = "",     # رمز من hf.co/settings/tokens بصلاحية Write — للرفع الدائم (الخلية 16)
    HF_REPO_ID     = "",     # فارغ = <اسم-حسابك>/rafidain-seedvc-iraqi تلقائياً
    HF_PRIVATE     = True,   # مستودع خاص لا يراه أحد غيرك
    HF_ENDPOINT    = "",     # مرآة HuggingFace إن لزم (مثال: https://hf-mirror.com) — اتركها فارغة عادة
    # تنظيف
    CLEAN_AFTER     = False, # حذف بيانات التدريب المفكوكة بعد التصدير (يوفر مساحة — الخلية 17)
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
#    CFG["STEPS"] = 1000
CFG_PATH.write_text(json.dumps(CFG, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- 5) المسارات الثابتة ----------
SEEDVC   = ROOT / "seed-vc"                    # مستودع Seed-VC (المدرب الأصلي)
RAFIDAIN = ROOT / "rafidain"                   # مستودع رافِدَين (الخدمة + خط البيانات)
DATA_DIR = ROOT / "data" / "iraqi_finetune"    # بيانات التدريب المجهزة (Opus 24kHz) — من الأرشيف الجاهز
RUN_DIR  = SEEDVC / "runs" / CFG["RUN_NAME"]   # نقاط التدريب والسجلات (محلي سريع)
EXPORTS  = ROOT / "exports"                    # حزم النموذج النهائية (محلي)
READY_TGZ = ROOT / "rafidain_current_sources_ready.tar.gz"   # الأرشيف الجاهز (محلي)
READY_STAMP = ROOT / ".ready_archive_ok"       # بصمة «الأرشيف مفكوك وسليم»
# ملاحظة: لا بيئة بايثون منفصلة في هذا الدفتر — كل المنطق يعمل داخل نواة
# الجلسة نفسها بالاستيراد المباشر؛ الحزم تُثبَّت في النواة (الخلية 5).

# مراياها على Drive (تُنشأ عند الحاجة — لا الآن إن لم يكن موصولاً)
def drive_run_dir():
    return (DRIVE_ROOT / "runs" / CFG["RUN_NAME"]) if (DRIVE_OK and DRIVE_ROOT) else None
def drive_export_base():
    return (DRIVE_ROOT / "exports") if (DRIVE_OK and DRIVE_ROOT) else None
def drive_data_dir():
    # نسخة الأرشيف على Drive (اختيارية — توفر إعادة التنزيل عند انقطاع الجلسة)
    return (DRIVE_ROOT / "data") if (DRIVE_OK and DRIVE_ROOT) else None

for p in (ROOT, EXPORTS):
    p.mkdir(parents=True, exist_ok=True)

# ---------- 6) بيئة الأوامر ----------
# (متغيرات البيئة فقط هنا — أداة أوامر النظام sh() تُعرَّف في الخلية 4،
#  ومنطق المشروع كله يُستدعى بالاستيراد المباشر من الخلية 6 وما بعدها)
ENV = dict(os.environ)
ENV["PYTHONUNBUFFERED"] = "1"
ENV["HF_HUB_DISABLE_XET"] = "1"           # تنزيل HTTP قابل للاستئناف — Xet يقيّد IPs Colab (خيار الخط الرسمي)
ENV["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
if CFG["HF_ENDPOINT"]:
    ENV["HF_ENDPOINT"] = CFG["HF_ENDPOINT"]

# ---------- 7) ملخص ----------
print("=" * 64)
print(f"جذر العمل (محلي سريع) : {ROOT}")
print(f"التخزين الدائم (Drive) : {DRIVE_ROOT if DRIVE_OK else 'غير موصول — اعتمد على الخلية 16 لاحقاً'}")
print("=" * 64)
print(f"  • البيانات          : أرشيف جاهز {CFG['READY_ARCHIVE_SIZE']/1e9:.2f}GB (بلا إعادة معالجة)")
print(f"  • التدريب            : {CFG['STEPS']} خطوة، حفظ كل {CFG['SAVE_EVERY']}, دفعة {CFG['BATCH']}")
print(f"  • اسم التشغيل        : {CFG['RUN_NAME']}")
print(f"  • نسخ Drive          : {'لحظي عند كل حفظ (آخر %d نقاط)' % CFG['DRIVE_CKPT_KEEP'] if (DRIVE_OK and CFG['DRIVE_BACKUP']) else ('غير متاح (Drive غير موصول)' if not DRIVE_OK else 'معطّل')}")
print(f"  • رمز HuggingFace    : {'مضبوط (سيُرفع للأبد في الخلية 16)' if CFG['HF_TOKEN'] else 'غير مضبوط (اختياري — الرفع الدائم)'}")
print(f"\n[المسارات]  البيانات: {DATA_DIR}")
print(f"            التدريب : {RUN_DIR}")
print(f"            التصدير : {EXPORTS}")
print(f"\nملاحظة: الرمز السري (إن وضعته) يُحفظ في {CFG_PATH} — على Drive الخاص بك فقط.")'''))

# ============================================================ CODE 3 — env check
CELLS_A.append(("code", r'''# ============================================================ #
#  الخلية ٣ — فحص البيئة: GPU / قرص / أدوات                    #
#  (يحدد حجم الدفعة تلقائياً حسب ذاكرة GPU المكتشفة)           #
#  (nvidia-smi استعلام نظام للقراءة فقط — عبر sh())           #
# ============================================================ #
import os, shutil, re, json, sys
from pathlib import Path

def gpu_query():
    try:
        # استعلام nvidia-smi عبر مُشغّل أوامر النظام (قراءة معلومات العتاد فقط)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            outf = Path(td) / "gpu.txt"
            sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits", "-f", str(outf)],
               check=False, echo=False)
            out = outf.read_text().strip().splitlines() if outf.exists() else []
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

# ---- القرص المحلي (الأرشيف الجاهز أنقص الحاجة كثيراً) ----
du = shutil.disk_usage(ROOT)
need = 14  # GB: أرشيف مفكوك + بيئة + نقاط أساسية + نقاط تدريب
print(f"[قرص محلي] المتاح على {ROOT}: حر {du.free/1e9:.1f}GB من {du.total/1e9:.1f}GB")
if du.free / 1e9 < need + 2:
    print(f"          [تحذير] المسار يحتاج ~{need}GB — حرر مساحة أو شغّل خلية التنظيف (17)")
else:
    print(f"          [جيد] يكفي لإعدادك الحالي (~{need}GB مطلوبة تقريباً — الأرشيف الجاهز وفّر نصف حاجة النسخة السابقة)")

# ---- Drive ----
if DRIVE_OK and DRIVE_ROOT:
    try:
        ddu = shutil.disk_usage(DRIVE_ROOT)
        print(f"[Drive] الحصة: حر {ddu.free/1e9:.1f}GB من {ddu.total/1e9:.1f}GB — النسخ الاحتياطية تحتاج ~2GB + الأرشيف 4.9GB إن اخترت حفظه")
    except Exception:
        print("[Drive] موصول (تعذر قراءة الحصة)")

# ---- الأدوات ----
for tool in ("ffmpeg", "ffprobe", "git", "curl"):
    print(f"[أداة] {tool:<8}: {'متوفر' if shutil.which(tool) else 'غير متوفر — تثبّته الخلية 4'}")

print("\n[فحص مبدئي] أكمل إلى الخلية 4 (التبعيات).")'''))

# ============================================================ CODE 4 — system deps + sh()
CELLS_A.append(("code", r'''# ============================================================ #
#  الخلية ٤ — أدوات النظام + مُشغّل أوامر النظام sh()          #
#  (apt فقط للأدوات: ffmpeg/git — وsh() تُستخدم لاحقاً لأوامر #
#   النظام الضرورية فقط: apt-get / git / tar / curl / pip)    #
#  Colab يعمل بصلاحيات root عادة — بلا sudo                   #
#  هذه الخلية آخر موضع لأوامر النظام: كل منطق المشروع بعد    #
#  الخلية 5 يعمل داخل النواة بالاستيراد المباشر — بلا أي     #
#  عملية ابن للتدريب أو الاستدلال أو خط البيانات             #
# ============================================================ #
import shutil, subprocess
from pathlib import Path

def have(x):
    return shutil.which(x) is not None

def sh(cmd, cwd=None, env=None, check=True, echo=True):
    # مُشغّل أوامر النظام: يشغّل أداة نظام (apt/git/tar/curl/pip) ويطبع
    # مخرجاتها لحظياً داخل الخلية ويعيد رمز الخروج.
    # ملاحظة سياسة الاستخدام: تُستخدم حصرياً لأدوات النظام أعلاه — أما التدريب
    # والاستدلال وخط البيانات فتعمل داخل نواة الدفتر بالاستيراد المباشر.
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

missing = [p for p in ("ffmpeg", "git") if not have(p)]
if missing:
    print(f"تثبيت حزم النظام: {', '.join(missing)} ...")
    sh(["apt-get", "update", "-qq"], check=False)
    sh(["apt-get", "install", "-y", "-qq", *missing])
for tool in ("ffmpeg", "git", "curl", "tar"):
    print(f"  {tool:<8}: {'متوفر' if have(tool) else 'غير متوفر!'}")
print("\n[جاهز] أدوات النظام كاملة ومُشغّل أوامر النظام معرّف — لا بيئة بايثون منفصلة في هذا الدفتر.")'''))

# ============================================================ CODE 5 — in-kernel packages
CELLS_A.append(("code", r'''# ============================================================ #
#  الخلية ٥ — حزم بايثون داخل نواة الجلسة نفسها               #
#  لا بيئة منفصلة ولا عملية ابن لمنطق المشروع: التدريب       #
#  وخط البيانات ستعمل بالاستيراد داخل هذه النواة مباشرة،      #
#  لذا نثبّت الإصدارات المطلوبة هنا في نواة Colab ونضبطها     #
#  (نفس الإصدارات التي يثبتها الخط الرسمي seed_vc_iraqi_colab.sh)
# ============================================================ #
import importlib, sys
from importlib.metadata import version as pkg_version, PackageNotFoundError

def have_pkg(name):
    try:
        pkg_version(name)
        return True
    except PackageNotFoundError:
        return False

def pip_install(pkgs):
    # تثبيت الحزم في نواة الجلسة (العملية نفسها التي تدير الدفتر)
    rc = sh([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
    return rc

def pkg_ok(spec, import_name):
    """spec مثل 'torch==2.4.0' — يتحقق أن الإصدار المطلوب مستورد فعلاً في النواة."""
    want = spec.split("==")[1] if "==" in spec else None
    try:
        mod = importlib.import_module(import_name)
    except Exception:
        return False
    if want is None:
        return True
    return getattr(mod, "__version__", "?").startswith(want)

# ---- 1) PyTorch 2.4.0 cu121 — الإصدار الذي ينجح به الخط الرسمي (نحو 2.5GB) ----
if not pkg_ok("torch==2.4.0", "torch"):
    print("تثبيت PyTorch 2.4.0 + torchaudio (cu121) في نواة الجلسة — عدة دقائق ...")
    idx = ["--index-url", "https://download.pytorch.org/whl/cu121"]
    rc = pip_install(["torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0", *idx])
    if rc != 0:
        pip_install(["torch", "torchvision", "torchaudio", *idx])
    # بعد التثبيت: يجب إعادة تشغيل النواة لتفعيل الإصدار الجديد (torch محمّل مسبقاً)
    print("""
[مهم] ثُبّت PyTorch 2.4.0 — أعد تشغيل النواة الآن ثم أكمل الخلايا:
      Runtime <- Restart session (أو Restart and run all)
      (Colab يحمّل torch بإصدار قديم مسبقاً؛ إعادة التشغيل شرط لتفعيل 2.4.0 داخل النواة)
      الخلايا كلها استئنافية — لا شيء يُعاد تنزيله بعد إعادة التشغيل.""")
else:
    import torch
    print(f"PyTorch {torch.__version__} جاهز في النواة | CUDA: {torch.cuda.is_available()}")

# ---- 2) بقية الاعتماديات (نفس إصدارات الخط الرسمي، بالاستيراد داخل النواة) ----
need_imports = {
    "librosa": "librosa==0.10.2", "soundfile": "soundfile==0.12.1",
    "munch": "munch==4.0.0", "einops": "einops==0.8.0",
    "transformers": "transformers==4.46.3", "pyarrow": "pyarrow==17.0.0",
    "yaml": "pyyaml", "huggingface_hub": "huggingface_hub",
    "matplotlib": "matplotlib", "scipy": "scipy==1.13.1", "numpy": "numpy==1.26.4",
}
missing = []
for imp, spec in need_imports.items():
    try:
        importlib.import_module(imp)
    except Exception:
        missing.append(spec)
if missing:
    print(f"تثبيت الاعتماديات الناقصة داخل النواة: {len(missing)} حزمة ...")
    rc = pip_install(missing)
    if rc != 0:
        pip_install([s.split("==")[0] for s in missing])
else:
    print("الاعتماديات موجودة أصلاً في النواة — تخطّي التثبيت.")

# ---- 3) تحقق نهائي داخل النواة (استيراد فعلي — لا فحص في عملية خارجية) ----
import torch, torchaudio
assert torch.__version__.startswith("2.4."), (
    f"النواة تحمّل torch {torch.__version__} وليس 2.4.0 — أعد تشغيل النواة "
    "(Runtime <- Restart session) ثم Run all؛ الخلايا استئنافية")
print(f"\n[جاهز] النواة نفسها هي بيئة التشغيل: torch {torch.__version__} | "
      f"torchaudio {torchaudio.__version__} | CUDA {torch.cuda.is_available()}")
print("        التدريب والاستدلال سيعملان بالاستيراد المباشر داخل هذه النواة — بلا أي عملية ابن.")'''))

# ============================================================ CODE 6 — clone repos + import modules
CELLS_A.append(("code", r'''# ============================================================ #
#  الخلية ٦ — استنساخ المستودعات ثم استيراد وحداتها مباشرة     #
#  (كل منطق المشروع يُستخدم من الآن بالاستيراد — بلا سطر       #
#   أوامر لخط البيانات أو التدريب أو الاستدلال)               #
# ============================================================ #
import sys, importlib
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
PIPE_SRC = RAFIDAIN / "scripts" / "data_research" / "iraqi_pipeline_v3.py"
assert PIPE_SRC.exists(), "iraqi_pipeline_v3.py مفقود في rafidain"
print(f"[جاهز] seed-vc   : {SEEDVC}")
print(f"[جاهز] rafidain  : {RAFIDAIN}")

# ---- استيراد خط بيانات العراق من المستودع المستنسخ (مصدر واحد للحقيقة) ----
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("iraqi_pipeline", PIPE_SRC)
PIPELINE = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(PIPELINE)
print(f"[استيراد] iraqi_pipeline: {len(PIPELINE.CLIPS):,} مقطعاً / {PIPELINE.TOTAL_CLIP_HOURS:.0f} ساعة في الكتالوج")

# وحدة مزامنة النقاط من رافِدَين (restore/push_once — بلا خيوط خلفية)
sys.path.insert(0, str(RAFIDAIN))
from finetune import checkpoint_sync as CKPT_SYNC
print(f"[استيراد] finetune.checkpoint_sync: {CKPT_SYNC.__name__}")

# ---- مدرب واستدلال Seed-VC: استيراد مباشر داخل نواة الدفتر ----
# train.py وinference.py يفتحان configs/hifigan.yml و./checkpoints نسبياً إلى cwd،
# لذلك نجعل مجلد seed-vc هو دليل عمل النواة منذ الآن (وتبقى مساراتنا كلها مطلقة).
import os as _os
_os.chdir(SEEDVC)
sys.path.insert(0, str(SEEDVC))
import train as SEED_TRAIN          # المدرب الرسمي: SEED_TRAIN.Trainer(...)
import inference as SEED_INFER      # الاستدلال الرسمي: SEED_INFER.main(args-like)
print(f"[استيراد] seed-vc train.Trainer + inference.main — داخل نواة الدفتر مباشرة")

SEEDVC_CFG = SEEDVC / "configs" / "presets" / "config_dit_mel_seed_uvit_xlsr_tiny.yml"
print(f"[مسار] إعداد التدريب الرسمي: {SEEDVC_CFG}")
print(f"[نواة] دليل العمل الحالي: {Path.cwd()} (seed-vc — لتصح المسارات النسبية للمدرب)")
print("\n[جاهز] المستودعات مستنسخة والوحدات مستوردة داخل النواة — تابع إلى خلية البيانات الجاهزة (7).")'''))
