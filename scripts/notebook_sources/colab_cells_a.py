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
| 1. البيئة | توصيل Drive، فحص GPU/القرص، تثبيت ffmpeg + uv + PyTorch 2.4.0 (cu121) في بيئة معزولة |
| 2. المستودعات | استنساخ Seed-VC + رافِدَين من GitHub، ثم **استيراد وحداتهما مباشرة** (بلا سطر أوامر) |
| 3. البيانات الجاهزة | تنزيل أرشيف **`rafidain_current_sources_ready.tar.gz`** (4.89GB) — بيانات تدريب مقطّعة ومرمّزة Opus 24kHz جاهزة، **بلا إعادة تنزيل من المصادر ولا إعادة معالجة** |
| 4. التدريب | ضبط دقيق في **المقدمة داخل الخلية** عبر `Trainer` المستورد، مع مرآة نقاط فورية إلى Drive عند كل حفظ |
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
| 4 | تبعيات النظام | ffmpeg + git + uv | أقل من دقيقة |
| 5 | بيئة Python | venv مستقل 3.11 + PyTorch 2.4.0 cu121 + الاعتماديات | 3–8 دقائق |
| 6 | المستودعات + الاستيراد | استنساخ Seed-VC + رافِدَين ثم **استيراد** خط البيانات والمدرب والاستدلال | أقل من دقيقة |
| 7 | البيانات الجاهزة | تنزيل أرشيف `rafidain_current_sources_ready.tar.gz` وفكّه — **بلا أي إعادة معالجة** | 5–20 دقيقة (مرة واحدة) |
| 8 | بياناتك الخاصة | (اختياري) مجلد لملفاتك الصوتية يُقطَّع ويُدمج في التدريب | حسبك |
| 9 | إحصاءات البيانات | تقرير الساعات/الملفات لكل مصدر (من فهرس الأرشيف) | ثوانٍ |
| 10 | نقاط النموذج | تنزيل DiT/XLSR/HiFT/CAMPPlus مسبقاً (نحو 2.2GB) — عبر استيراد خط البيانات | 2–10 دقائق |
| 11 | استعادة من Drive | (بعد أي انقطاع) إعادة آخر نقطة تدريب من Drive | ثوانٍ |
| 12 | التدريب (في المقدمة) | `Trainer` المستورد يعمل داخل الخلية + مرآة Drive عند كل حفظ | 2–4 ساعات (T4) |
| 13 | التحقق | استدلال حقيقي بـ `inference.py` المستورد + استماع داخل الدفتر | 1–3 دقائق |
| 14 | التصدير | حزمة أوزان + إعداد + تقرير → ونسخها إلى Drive | ثوانٍ |
| 15 | أرشيف tar.gz | ملف واحد في Drive جاهز للتنزيل | ثوانٍ |
| 16 | HuggingFace Hub | **التخزين السحابي الدائم** — النموذج في حسابك للأبد | 1–5 دقائق |
| 17 | تنظيف المساحة | (اختيارية) حذف الأرشيف المفكوك والبيئة المؤقتة | ثوانٍ |
| 18 | النشر والاستخدام | أوامر جاهزة: استدلال فوري + خدمة رافِدَين على نفس الجلسة | اختيارية |

**قواعد ذهبية**
- كل الخلايا **آمنة لإعادة التشغيل** — لا شيء يُعاد تنزيله أو تنفيذه إن كان منجزاً
  (بصمة الأرشيف + علامات `.done` + فحوص وجود).
- إذا انقطعت الجلسة: افتح الدفتر وشغّل **Run all** — البيئة تُبنى، والأرشيف يُفك
  فوراً من نسخة Drive إن وُجدت، والخلية 11 تستعيد نقاط التدريب، والتدريب يكمل من
  آخر نقطة حفظ.
- **لا توجد عمليات خلفية إطلاقاً** في هذا الدفتر — كل خلية تنجز عملها وتنتهي.
  هذا تحديداً هو ما يمنع قطع الجلسة «disallowed usage» في الطبقة المجانية.
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
import os, sys, json, subprocess, shutil, time, re
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
VENV_PY  = ROOT / "venv" / "bin" / "python"    # بايثون البيئة المعزولة (الخلية 5)
READY_TGZ = ROOT / "rafidain_current_sources_ready.tar.gz"   # الأرشيف الجاهز (محلي)
READY_STAMP = ROOT / ".ready_archive_ok"       # بصمة «الأرشيف مفكوك وسليم»

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

# ---------- 6) بيئة الأوامر وأداة تشغيل موحدة ----------
# (تُستخدم فقط لأوامر النظام الضرورية: apt-get / uv / git — أما منطق المشروع
#  كله فيُستدعى بالاستيراد المباشر من الخلية 6 وما بعدها)
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
# ملاحظة: نستورده بنواة النوت بوك (نفس المنطق الذي ستستورده به البيئة المعزولة لاحقاً)
# ليستخدمه التدريب؛ الاعتماديات الثقيلة (torch/transformers) غير مطلوبة لهذه الوحدة
# إلا في أوامر fetch — والأوامر التي نستدعيها هنا (prefetch/export/stats/verify)
# تعمل داخل البيئة المعزولة عبر runpy بنفس الوحدة.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("iraqi_pipeline", PIPE_SRC)
PIPELINE = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(PIPELINE)
print(f"[استيراد] iraqi_pipeline: {len(PIPELINE.CLIPS):,} مقطعاً / {PIPELINE.TOTAL_CLIP_HOURS:.0f} ساعة في الكتالوج")

# وحدة مزامنة النقاط من رافِدَين (restore/push_once — بلا خيوط خلفية)
sys.path.insert(0, str(RAFIDAIN))
from finetune import checkpoint_sync as CKPT_SYNC
print(f"[استيراد] finetune.checkpoint_sync: {CKPT_SYNC.__name__}")

# مدرب واستدلال Seed-VC: يستوردان في خلايا التدريب/التحقق داخل البيئة المعزولة
# (يحتاجان torch 2.4 + CUDA — نستوردهما هناك بالضبط قبل الاستخدام)
SEEDVC_CFG = SEEDVC / "configs" / "presets" / "config_dit_mel_seed_uvit_xlsr_tiny.yml"
print(f"[مسار] إعداد التدريب الرسمي: {SEEDVC_CFG}")
print("\n[جاهز] المستودعات مستنسخة والوحدات قابلة للاستيراد — تابع إلى خلية البيانات الجاهزة (7).")'''))
