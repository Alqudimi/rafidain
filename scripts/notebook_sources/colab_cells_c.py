# -*- coding: utf-8 -*-
"""Part C: verify/export markdown + code cells 17-19, persistence md + code 20-22, troubleshooting + summary md."""

CELLS_C = []

# ============================================================ MD 6 — verify & export
CELLS_C.append(("markdown", r'''<div dir="rtl">

## التحقق والتصدير

قبل الاحتفال بالنموذج نجري **استدلالاً حقيقياً** بالنقطة الناتجة: مقطع مصدر من أمثلة Seed-VC يُحوَّل
بصوت مرجعي عراقي من بياناتك المجهزة، وتستمع للناتج داخل الدفتر مباشرة. بعدها نبني **حزمة التصدير**
وننسخها فوراً إلى Drive:

```
exports/iraqi_v1/                          (محلي + MyDrive/rafidain_seedvc/exports/iraqi_v1/)
├── seed_uvit_tat_xlsr_tiny_iraqi.pth      الأوزان المضبوطة (للنشر المباشر)
├── config_dit_mel_seed_uvit_xlsr_tiny.yml
├── training_report.json                   تقرير التدريب (البيانات والكتالوج والتاريخ)
├── _stats.json                            ساعات/ملفات كل مصدر
├── train_console.log                      سجل التدريب كاملاً
├── loss_curve.png                         منحنى الخسارة
└── README.md                              أوامر الاستخدام الجاهزة
```

</div>'''))

# ============================================================ CODE 17 — verify
CELLS_C.append(("code", r'''# ============================================================ #
#  الخلية ١٧ — التحقق: استدلال حقيقي بالنقطة المضبوطة          #
#  (مصدر من أمثلة Seed-VC + مرجع عراقي من بياناتك -> تشغيل)    #
# ============================================================ #
from IPython.display import Audio, display
from pathlib import Path

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
if outs:
    print(f"\nناتج التحويل: {outs[0].name}  ({outs[0].stat().st_size/1e6:.1f} MB)")
    print("استمع للناتج (المصدر إنجليزي يحمل الطابع الصوتي العراقي المرجعي):")
    display(Audio(filename=str(outs[0])))
    for extra in outs[1:3]:
        print(f"ناتج إضافي: {extra.name}")
        display(Audio(filename=str(extra)))
else:
    print("لم يُنتج ملف صوتي — راجع السجل أعلاه. (غير قاتل: الأوزان ستصدَّر في الخلية 18 على أي حال)")'''))

# ============================================================ CODE 18 — export
CELLS_C.append(("code", r'''# ============================================================ #
#  الخلية ١٨ — تصدير الحزمة النهائية                           #
#  أوزان + إعداد + تقرير + سجلات — محلياً ثم نسخها إلى Drive   #
# ============================================================ #
import shutil
from pathlib import Path

FT = RUN_DIR / "ft_model.pth"
assert FT.exists(), "لا يوجد ft_model.pth — أكمل التدريب أولاً"
EXPORT_DIR = EXPORTS / CFG["RUN_NAME"]
CFG_TINY = SEEDVC / "configs" / "presets" / "config_dit_mel_seed_uvit_xlsr_tiny.yml"

sh([VENV_PY, PIPE, "export",
    "--run-dir", RUN_DIR,
    "--config", CFG_TINY,
    "--data-dir", DATA_DIR,
    "--out", EXPORT_DIR])

# توثيق إضافي يُنسخ مع الحزمة (قابلية إعادة الإنتاج جزء من الجودة)
for extra in (RUN_DIR / "train_console.log", RUN_DIR / "loss_curve.png",
              RUN_DIR / "config_dit_mel_seed_uvit_xlsr_tiny.yml", DATA_DIR / "_stats.json"):
    if extra.exists():
        shutil.copy2(extra, EXPORT_DIR / extra.name)
        print(f"  + {extra.name}")

total = 0
print(f"\nحزمة التصدير (محلي): {EXPORT_DIR}")
for p in sorted(EXPORT_DIR.rglob("*")):
    if p.is_file():
        s = p.stat().st_size
        total += s
        print(f"  {p.name:<46} {s/1e6:9.2f} MB")
print(f"  {'الإجمالي':<46} {total/1e6:9.2f} MB")

# ---- النسخ الفوري إلى Drive (ديمومة فورية) ----
DRV_EXP = (globals().get("DRIVE_ROOT") / "exports" / CFG["RUN_NAME"]) if (globals().get("DRIVE_OK") and globals().get("DRIVE_ROOT")) else None
if DRV_EXP:
    print(f"\nنسخ الحزمة إلى Drive: {DRV_EXP} ...")
    DRV_EXP.mkdir(parents=True, exist_ok=True)
    for p in sorted(EXPORT_DIR.rglob("*")):
        if p.is_file():
            shutil.copy2(p, DRV_EXP / p.name)
    print("[Drive] الحزمة محفوظة بشكل دائم — تنجو من أي انقطاع جلسة.")'''))

# ============================================================ CODE 19 — archive
CELLS_C.append(("code", r"""# ============================================================ #
#  الخلية ١٩ — أرشيف tar.gz واحد (في Drive + محلي)             #
# ============================================================ #
import shutil
from pathlib import Path

EXPORT_DIR = EXPORTS / CFG["RUN_NAME"]
assert (EXPORT_DIR / "seed_uvit_tat_xlsr_tiny_iraqi.pth").exists(), "شغّل خلية التصدير (18) أولاً"

base = EXPORTS / f"rafidain_seedvc_{CFG['RUN_NAME']}"
archive = shutil.make_archive(str(base), "gztar", root_dir=str(EXPORTS), base_dir=EXPORT_DIR.name)
size_mb = Path(archive).stat().st_size / 1e6
print(f"الأرشيف جاهز: {archive}  ({size_mb:.1f} MB)")

# ---- نسخه إلى Drive ----
if globals().get("DRIVE_OK") and globals().get("DRIVE_ROOT"):
    drv_arc = globals()["DRIVE_ROOT"] / "exports" / Path(archive).name
    shutil.copy2(archive, drv_arc)
    print(f"[Drive] نسخة دائمة: {drv_arc}")

print('''
لتنزيله إلى جهازك:
  • من Drive (الأسهل): drive.google.com <- مجلد rafidain_seedvc/exports <- نزّل الملف
  • أو من لوحة ملفات Colab (الشريط الجانبي): مجلد exports <- كليك يمين <- Download
  • أو أضف في خلية جديدة:  from google.colab import files; files.download("<مسار الأرشيف>")
''')"""))

# ============================================================ MD 7 — persistence layers
CELLS_C.append(("markdown", r'''<div dir="rtl">

## الحفظ الدائم — ثلاث طبقات (حتى تنزّلها بعد سنة)

| الطبقة | أين تبقى | كم تدوم | كيف تنزّلها لاحقاً |
|---|---|---|---|
| 1. **Google Drive** | `MyDrive/rafidain_seedvc/` — نسخ احتياطية أثناء التدريب + حزمة التصدير + الأرشيف | دائمة في حسابك مهما انقطعت جلسات Colab | من أي متصفح/جهاز عبر drive.google.com |
| 2. **HuggingFace Hub** | مستودع في حسابك (خاص افتراضياً) | **دائمة مجاناً — بلا علاقة بـ Google إطلاقاً** | `hf download <repo>` أو زر التنزيل من صفحة المستودع |
| 3. القرص المحلي `/content` | البيانات الخام + بيانات التدريب + البيئة | تضيع عند انقطاع/إعادة تعيين الجلسة | (قابلة لإعادة البناء بالخلايا 5-11 — كلها استئنافية) |

> **الوصية**: الطبقة 2 هي ضمانتك الحقيقية للمدى البعيد — حتى لو فقدت حساب Google أو حُذف Drive،
> يبقى النموذج في HuggingFace قابلاً للتنزيل من أي مكان في العالم. تحتاج فقط رمز API بصلاحية كتابة
> من `hf.co/settings/tokens` تضعه في الخلية 2 ثم تشغّل الخلية 20.

ما يصل إلى Drive **أوتوماتيكياً** أثناء التدريب (مراقب الخلية 14): آخر نقطتين `DiT_epoch_*.pth` +
`train_console.log` + `loss_curve.png` + `ft_model.pth` عند الاكتمال — كل ذلك في
`MyDrive/rafidain_seedvc/runs/<اسم-التشغيل>/`، بمساحة لا تتجاوز ~2GB.

</div>'''))

# ============================================================ CODE 20 — HF Hub upload
CELLS_C.append(("code", r"""# ============================================================ #
#  الخلية ٢٠ — الرفع الدائم إلى HuggingFace Hub                #
#  النموذج يُخزَّن في حسابك أنت، مجاناً، بلا علاقة بـ Google     #
#  (أعد تشغيلها في أي وقت مستقبلاً — ولو بعد شهور)              #
# ============================================================ #
from pathlib import Path

EXPORT_DIR = EXPORTS / CFG["RUN_NAME"]
HF_SCRIPT = ROOT / "hf_upload.py"
HF_SCRIPT.write_text(r'''
import os, time
from huggingface_hub import HfApi

token = os.environ["HF_TOKEN"]
export_dir = os.environ["NB_EXPORT_DIR"]
api = HfApi(token=token)
user = api.whoami()["name"]
repo_id = os.environ.get("NB_REPO_ID", "").strip() or f"{user}/rafidain-seedvc-iraqi"
private = os.environ.get("NB_PRIVATE", "1") == "1"
api.create_repo(repo_id, private=private, exist_ok=True)
print(f"[hf] upload {export_dir} -> {repo_id} (private = {private}) ...")
api.upload_folder(
    folder_path=export_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message=f"Rafidain Iraqi fine-tune ({time.strftime('%Y-%m-%d %H:%M')})",
)
print("[hf] REPO_URL: https://huggingface.co/" + repo_id)
''', encoding="utf-8")

if not CFG["HF_TOKEN"]:
    print("لا يوجد رمز HuggingFace بعد — لا مشكلة: كل أصولك محفوظة على Drive،")
    print("ويمكنك تنفيذ الرفع الدائم في أي وقت لاحق (الخلية تعمل مستقلة):\n")
    print("  1) أنشئ حساباً مجانياً في huggingface.co إن لم يكن لديك")
    print("  2) من hf.co/settings/tokens أنشئ New Token بصلاحية Write وانسخه")
    print('  3) في الخلية 2 اضبط:  CFG["HF_TOKEN"] = "hf_ضع-الرمز-هنا"  ثم أعد تشغيلها')
    print("  4) أعد تشغيل هذه الخلية — سترفع الحزمة كاملة إلى مستودع خاص في حسابك")
else:
    assert (EXPORT_DIR / "seed_uvit_tat_xlsr_tiny_iraqi.pth").exists(), "شغّل خلية التصدير (18) أولاً"
    env = dict(ENV)
    env["HF_TOKEN"] = str(CFG["HF_TOKEN"])
    env["NB_EXPORT_DIR"] = str(EXPORT_DIR)
    env["NB_REPO_ID"] = str(CFG.get("HF_REPO_ID", ""))
    env["NB_PRIVATE"] = "1" if CFG["HF_PRIVATE"] else "0"
    sh([VENV_PY, HF_SCRIPT], env=env)
    print('''
[تم] النموذج الآن في حسابك على HuggingFace — بشكل دائم.
لتنزيله في أي وقت مستقبلاً من أي جهاز (بلا Google ولا Colab):
    pip install -U "huggingface_hub[cli]"
    hf download <اسم-حسابك>/rafidain-seedvc-iraqi --local-dir ./mymodel
أو من بايثون:
    from huggingface_hub import snapshot_download
    snapshot_download("<اسم-حسابك>/rafidain-seedvc-iraqi", local_dir="./mymodel")
أو مباشرة من صفحة المستودع في المتصفح.''')"""))

# ============================================================ CODE 21 — cleanup
CELLS_C.append(("code", r'''# ============================================================ #
#  الخلية ٢١ — (اختيارية) تحرير مساحة القرص المحلي             #
#  تحذف المصادر الخام القابلة لإعادة الجلب فقط:                #
#    raw/yodas_webm + مخابئ parquet — وتبقى بيانات التدريب     #
#    المجهزة وكل النقاط والحزم كما هي                          #
#  ملاحظة: علامات .done تبقى، فلن يُعاد التنزيل تلقائياً؛       #
#  لإعادة الجلب مستقبلاً احذف ملف .done المناسب من raw/.done/  #
# ============================================================ #
import shutil
from pathlib import Path

UV = globals().get("UV") or shutil.which("uv")

def dir_gb(p: Path) -> float:
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9

targets = [ROOT / "raw" / "yodas_webm", ROOT / "raw" / "_parquet", ROOT / "raw" / "_safaa_src"]
print(f"المساحة الحرة الآن: {shutil.disk_usage(ROOT).free/1e9:.1f} GB")
for t in targets:
    s = dir_gb(t)
    print(f"  {t.name:<12} {s:6.2f} GB" + ("   (قابل لإعادة الجلب بالخليتين 8 و9)" if s > 0 else "   (فارغ)"))

if CFG["CLEAN_RAW_AFTER"]:
    for t in targets:
        if t.exists():
            shutil.rmtree(t)
            print(f"  حُذف: {t.name}")
    if UV:
        sh([UV, "cache", "clean"], check=False)
    else:
        sh([str(VENV_PY), "-m", "pip", "cache", "purge"], check=False)
    print(f"المساحة الحرة بعد التنظيف: {shutil.disk_usage(ROOT).free/1e9:.1f} GB")
    print("بيانات التدريب الجاهزة في data/iraqi_finetune — لا حاجة للمصادر الخام إلا لإعادة التجهيز.")
else:
    print('\nللتفعيل: اجعل  CFG["CLEAN_RAW_AFTER"] = True  في الخلية 2 ثم أعد تشغيل هذه الخلية.')'''))

# ============================================================ CODE 22 — deployment
CELLS_C.append(("code", r"""# ============================================================ #
#  الخلية ٢٢ — النشر: كيف تستخدم النموذج المضبوط؟              #
#  (خلية إرشادية تطبع الأوامر الجاهزة بمساراتك الفعلية)        #
# ============================================================ #
from pathlib import Path

EXPORT_DIR = EXPORTS / CFG["RUN_NAME"]
W = EXPORT_DIR / "seed_uvit_tat_xlsr_tiny_iraqi.pth"
C = EXPORT_DIR / "config_dit_mel_seed_uvit_xlsr_tiny.yml"

if not W.exists():
    print("شغّل خلية التصدير (18) أولاً حتى تتوفر الحزمة.")
else:
    print(f'''
==========================================================
 نشر النموذج المضبوط — ثلاثة مسارات جاهزة
==========================================================

(أ) استدلال فوري في هذه الجلسة (البيئة جاهزة الآن):
    cd {SEEDVC}
    {VENV_PY} inference.py \\
        --checkpoint {W} \\
        --config     {C} \\
        --source <صوت-المتكلم-الأصلي.wav> \\
        --target  <صوت-مرجعي-عراقي.wav> \\
        --output  ./out

(ب) خدمة رافِدَين الكاملة على نفس جلسة Colab — REST + WebSocket + واجهة ويب
    بنفق عام مجاني (trycloudflare) للاستنساخ اللحظي من أي مكان.
    المستودع منسوخ هنا: {RAFIDAIN}
    في خلية جديدة الصق ما يلي (توجيه الخدمة إلى أوزانك المضبوطة):

      import os
      os.environ["RAFIDAIN__ENGINE__LOCAL_CHECKPOINT"] = r"{W}"
      os.environ["RAFIDAIN__ENGINE__LOCAL_CONFIG"]     = r"{C}"
      os.environ["RAFIDAIN__ENGINE__DEVICE"]           = "cuda"
      os.environ["RAFIDAIN__API_KEYS"]                 = r'["<مفتاح-API-عشوائي-خاص-بك>"]'
      get_ipython().system("bash {RAFIDAIN}/scripts/install/setup_colab.sh")

    بعد دقائق يطبع رابطاً عاماً مؤقتاً:
      لوحة التحكم:  <الرابط>/web   |   البث الحي:  <الرابط>/live   |   API:  <الرابط>/docs
    (متغيرا البيئة LOCAL_CHECKPOINT / LOCAL_CONFIG هما المساران أعلاه — يُقرآن مباشرة من البيئة)

(ج) على أي جهاز GPU آخر (خادمك / RunPod / غيره):
    1) نزّل الحزمة من Drive (rafidain_seedvc/exports) أو من HuggingFace (الخلية 20)
    2) git clone https://github.com/Plachtaa/seed-vc && cd seed-vc
    3) python inference.py --checkpoint <الحزمة>/seed_uvit_tat_xlsr_tiny_iraqi.pth \\
           --config <الحزمة>/config_dit_mel_seed_uvit_xlsr_tiny.yml \\
           --source a.wav --target ref.wav --output ./out
    أو خدمة رافِدَين الكاملة بنفس متغيرَي البيئة في (ب) — الدليل: DEPLOYMENT.md و WEBUI.md

تذكير أخلاقي: استنساخ صوت شخص حقيقي يتطلب موافقته الصريحة الموثقة،
وأفصح للطرف الآخر عند ما يوجبه القانون — راجع قسم الأمان في README المستودع.
''')"""))

# ============================================================ MD 8 — troubleshooting
CELLS_C.append(("markdown", r'''<div dir="rtl">

## استكشاف الأخطاء

| المشكلة | الحل |
|---|---|
| انقطعت الجلسة / أُعيد التشغيل | افتح الدفتر ← **Run all**: البيئة تُبنى، الخلية 13 تستعيد النقاط من Drive، التدريب يكمل من آخر نقطة (الخسارة ≤ 250 خطوة). البيانات الخام إن فُقدت تُجلب مجدداً بالخليتين 8-9 (استئناف كامل) |
| «No GPU available» | Runtime ← Change runtime type ← Hardware accelerator: GPU ← Save، ثم أعد تشغيل الخلايا بالترتيب |
| `CUDA out of memory` أثناء التدريب | خفّض الدفعة: `CFG["BATCH"] = 1` (أو 2) في الخلية 2، أعد تشغيلها ثم أعد الخلية 14 — يستأنف من آخر نقطة |
| انقطعت الجلسة وأنا بعيد عن المتصفح | طبيعي بعد ~90 دقيقة خمول — شغّل خلية 16 في المرة القادمة (تبقي النواة نشطة)؛ نقاطك على Drive على أي حال |
| نفدت مساحة Drive | المراقب يُبقي آخر نقطتين فقط (~1GB)؛ احذف مجلدات runs قديمة من Drive، أو اعتمد على HF Hub (الخلية 20) كطبقة وحيدة |
| نفد القرص المحلي `/content` | شغّل خلية التنظيف (21)، أو قلّل `YODAS_HOURS` وأعد الخليتين 9 و11 (الموجود يُتخطى) |
| Rate limit من HuggingFace (429) | Xet معطّل أصلاً هنا؛ انتظر دقائق وأعد المحاولة، أو اضبط `CFG["HF_ENDPOINT"] = "https://hf-mirror.com"` في الخلية 2 |
| بطء تنزيل YODAS | ارفع `FETCH_WORKERS` إلى 32 في الخلية 2 ثم أعد الخلية 9 — يستأنف دون إعادة ما نُزّل |
| `too few training clips` في الخلية 11 | تأكد أن الخليتين 8 و9 أنزلتا بيانات (انظر مجلد `raw/`) ثم أعد المحاولة |
| فشلت خلية التحقق (17) | غير قاتلة — الأوزان تصدَّر على أي حال؛ جرّب أمر inference.py اليدوي المطبوع في الخلية 22 |
| فشل استيراد torch داخل البيئة | احذف مجلد `ROOT/venv` وأعد الخلية 5 (إعادة بناء كاملة ~5 دقائق) |
| أعدت تشغيل النواة وضاعت المتغيرات | شغّل الخلايا بالترتيب من 1 — الإعدادات محفوظة (على Drive إن كان موصولاً) وتعود كما كانت |
| أريد تجربة نقطة وسيطة | ملفات `DiT_epoch_*_step_*.pth` (محلياً أو على Drive) تُقبل مباشرة في `inference.py --checkpoint` |

</div>'''))

# ============================================================ MD 9 — final summary
CELLS_C.append(("markdown", r'''<div dir="rtl">

## الخلاصة — أين كل شيء؟ (خريطة المقتنيات الدائمة)

| الأصل | المسار الدائم |
|---|---|
| حزمة النموذج النهائية | **Google Drive:** `MyDrive/rafidain_seedvc/exports/<اسم-التشغيل>/` |
| أرشيف النموذج (ملف واحد) | **Google Drive:** `MyDrive/rafidain_seedvc/exports/rafidain_seedvc_<اسم-التشغيل>.tar.gz` |
| نسخة HuggingFace الدائمة | `https://huggingface.co/<حسابك>/rafidain-seedvc-iraqi` (الخلية 20) |
| نقاط التدريب الاحتياطية + السجلات + منحنى الخسارة | **Google Drive:** `MyDrive/rafidain_seedvc/runs/<اسم-التشغيل>/` (تحديث تلقائي أثناء التدريب) |
| الإعدادات | **Google Drive:** `MyDrive/rafidain_seedvc/nb_config.json` |
| بيانات التدريب المجهزة | محلي `/content/iraqi_seedvc/data/iraqi_finetune/` — تُبنى من جديد عند الحاجة (استئناف كامل) |
| البيئة + المستودعات + الخام | محلي `/content/iraqi_seedvc/` — تُبنى من جديد بالخلايا 4-9 |

**لتنزيل النموذج في أي وقت مستقبلاً** — الطبقة الأضمن (HuggingFace، بلا Google نهائياً):

```bash
pip install -U "huggingface_hub[cli]"
hf download <اسم-حسابك>/rafidain-seedvc-iraqi --local-dir ./mymodel
```

ثم تشغيله فوراً على أي جهاز فيه GPU:

```bash
git clone https://github.com/Plachtaa/seed-vc && cd seed-vc
python inference.py \
    --checkpoint ./mymodel/seed_uvit_tat_xlsr_tiny_iraqi.pth \
    --config     ./mymodel/config_dit_mel_seed_uvit_xlsr_tiny.yml \
    --source a.wav --target ref.wav --output ./out
```

أو من Drive مباشرة: نزّل مجلد `rafidain_seedvc/exports/<اسم-التشغيل>/` — فيه نفس الملفات
مع README أوامر الاستخدام و`training_report.json` التوثيقي.

وللاستنساخ اللحظي الكامل (REST + WebSocket + واجهة ويب بنفق عام): خلية النشر (22) المسار (ب)
تُشغّل خدمة رافِدَين كاملة على نفس جلسة Colab بأوزانك المضبوطة خلال دقائق.

</div>'''))
