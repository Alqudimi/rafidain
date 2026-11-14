# -*- coding: utf-8 -*-
"""Part C: training markdown + code cells 11-12 (restore, foreground training),
verify/export/persist code cells 13-18, troubleshooting + summary md.

v2: training runs in the FOREGROUND of its cell by importing seed-vc's train.py
as a module and calling Trainer(...).train() directly — no Popen, no
start_new_session, no PID files, no monitor/wait loops (the patterns that got
the free Colab tier to kill the runtime). A context-managed wrapper around
torch.save mirrors every checkpoint to Drive synchronously from inside the
training process itself, and a tee of stdout keeps a full console log.
Verify uses inference.py's load_models/main imported in the venv python.
"""

CELLS_C = []

# ============================================================ MD 5 — training explanation
CELLS_C.append(("markdown", r'''<div dir="rtl">

## مرحلة التدريب — داخل نواة الدفتر، بلا عمليات خلفية وبلا subprocess

- **نقطة البداية دائماً** الأوزان المنشورة `DiT_uvit_tat_xlsr_ema.pth` من `Plachta/Seed-VC`
  (لا تدريب من الصفر أبداً). إن وُجدت نقطة سابقة في مجلد التشغيل يُستأنف منها تلقائياً
  (سلوك المدرب الرسمي: يبحث عن أحدث `DiT_epoch_*_step_*.pth` ويكمل منها، ويُبقي آخر
  نقطتين فقط).
- **كيف يعمل هنا؟** `import train` من مستودع Seed-VC المستنسخ (تمت في الخلية 6)، ثم
  ننشئ `Trainer` ونستدعي `.train()` **داخل نواة الدفتر نفسها** — لا بيئة منفصلة ولا عملية
  ابن ولا سطر أوامر إطلاقاً. ترى شريط تقدم tqdm وأسطر `epoch E, step S, loss: L`
  تظهر مباشرة أمامك. هذا هو النمط الذي تسمحه سياسة Colab المجانية: استخدام تفاعلي
  حقيقي للنواة.
- **لماذا لا عملية خلفية ولا subprocess؟** النسخة الأولى كانت تطلق التدريب كعملية
  مستقلة تعيش بعد انتهاء الخلية، مع مراقب نسخ دائم وحلقة انتظار لساعات — وهذا
  يُصنَّف «استخداماً غير مسموح» في الطبقة المجانية فتُقطع الجلسة فجأة
  (Runtime disconnected). النسخة الثانية أزالت الخلفية لكن أبقت تشغيل بايثون بعملية
  ابن؛ هذه النسخة تشغّل المدرب نفسه داخل النواة — صفر عمليات خارجية لمنطق المشروع.
- **النسخ إلى Drive رغم ذلك — كيف؟** نغلّف `torch.save` (التي يستدعيها المدرب عند
  كل نقطة حفظ) بغلاف ينسخ النقطة الجديدة إلى Drive **فور حفظها من داخل النواة
  نفسها** — بلا مراقب ولا خيط خلفي. أقصى خسارة عند أي انقطاع = الخطوات منذ
  آخر `SAVE_EVERY` (250 افتراضياً ≈ دقائق معدودة على T4).
- **مخرجات التدريب**: سطر خسارة كل 10 خطوات، نقطة كل 250 خطوة (تُنسخ فوراً)، وفي
  النهاية `ft_model.pth` (أوزان DiT المضبوطة — كل ما يلزم للنشر والاستدلال) ثم تُنسخ
  هي أيضاً إلى Drive مع السجل الكامل.
- **الإعدادات الموصى بها من دليل المستودع (FINETUNING.md)**: دفعة 2 لذاكرة 16GB (T4)
  و4 لـ 24GB (L4)؛ 200–500 خطوة لمتحدث واحد؛ **1500–4000 خطوة لتكيّف عام مع اللهجة**
  (3000 افتراضياً هنا)؛ معدل التعلم مثبّت في الإعداد الرسمي (1e-5) — لا ترفعه لنموذج
  25M باراميتر.
- **انقطعت الجلسة؟** افتح الدفتر ← **Run all**: تُثبَّت الحزم في النواة، والأرشيف
  الجاهز يُفك من نسخة Drive إن وُجدت، والخلية 11 تستعيد أحدث نقطة من Drive، ويستأنف التدريب من
  عندها تلقائياً.

مدة تقديرية: 3000 خطوة على **T4 في حدود 2–4 ساعات تقريباً** (حسب أطوال المقاطع)، وعلى
**L4 نحو نصف ذلك**. شريط التقدم يظهر بعد أول دقيقة فتعرف سرعتك الفعلية بدقة.

> **تنبيه مهم**: أبقِ هذا التبويب مفتوحاً أثناء خلية التدريب — إغلاق التبويب قد
> يُوقف الجلسة (سلوك Colab المعروف). النقاط محفوظة على Drive أياً حدث، لذا حتى لو
> انقطعت الجلسة فجأة: Run all وتكمل من آخر نقطة.

</div>'''))

# ============================================================ CODE 11 — restore from Drive
CELLS_C.append(("code", r'''# ============================================================ #
#  الخلية ١١ — استعادة نقاط التدريب من Drive (بعد أي انقطاع)   #
#  تعيد أحدث نقاط التفتيش + السجلات من نسخة Drive الاحتياطية   #
#  إلى القرص المحلي فيستأنف التدريب من حيث توقف                #
#  (خلية استباقية: لا تفعل شيئاً إن كان المحلي أحدث أو لا نسخة)#
#  عبر وحدة finetune.checkpoint_sync المستوردة — بلا سطر أوامر #
# ============================================================ #
import shutil, re
from pathlib import Path

drv = drive_run_dir()

# النقاط المحلية الحالية
loc_ckpts = sorted(RUN_DIR.glob("DiT_epoch_*_step_*.pth"), key=CKPT_SYNC._step_of)
if loc_ckpts or (RUN_DIR / "ft_model.pth").exists():
    last = f" (حتى الخطوة {CKPT_SYNC._step_of(loc_ckpts[-1])})" if loc_ckpts else ""
    print(f"[محلي] توجد نقاط محلية أصلاً{last} — لا حاجة للاستعادة.")
    print("        (لفرض الاستعادة من Drive احذف مجلد التشغيل المحلي ثم أعد هذه الخلية)")
elif drv and Path(drv).exists():
    # استعادة أحدث نقطة من Drive (نفس منطق السكربت الرسمي — عبر الوحدة المستوردة)
    restored = CKPT_SYNC.restore(RUN_DIR, Path(drv))
    if restored is None:
        print(f"[Drive] لا توجد نسخة احتياطية سابقة في {drv} — تشغيل جديد كلياً.")
    else:
        ck = sorted(RUN_DIR.glob("DiT_epoch_*_step_*.pth"), key=CKPT_SYNC._step_of)
        if ck:
            print(f"\n[تم] أُعيدت أحدث نقطة إلى الخطوة {CKPT_SYNC._step_of(ck[-1])} — التدريب سيكمل منها (الخلية 12).")
        elif (RUN_DIR / "ft_model.pth").exists():
            print("\n[تم] النموذج النهائي ft_model.pth مستعاد — يمكنك القفز إلى التحقق (الخلية 13).")
        else:
            print("\n[تم] استُعيدت السجلات فقط — سيبدأ التدريب من الصفر.")
else:
    print("[استعادة] لا نسخة Drive (غير موصول أو لا سجل سابق) — تشغيل جديد كلياً.")
    print("          (التدريب يبدأ من الأوزان المنشورة DiT_uvit_tat_xlsr_ema.pth — هذا طبيعي)")'''))

# ============================================================ CODE 12 — in-kernel training
CELLS_C.append(("code", r'''# ============================================================ #
#  الخلية ١٢ — التدريب: داخل نواة الدفتر نفسها                 #
#  SEED_TRAIN.Trainer (المستورد في الخلية 6) يُنشأ ويُشغَّل      #
#  هنا مباشرة — بلا عملية ابن ولا سطر أوامر إطلاقاً         #
#  كل نقطة حفظ تُنسخ فوراً إلى Drive من داخل النواة (غلاف      #
#  torch.save) — والسجل الكامل يُحفظ في train_console.log     #
# ============================================================ #
import os, sys, time, json, glob, re, shutil, contextlib, io
from pathlib import Path
import torch

FT = RUN_DIR / "ft_model.pth"

if FT.exists():
    print(f"النموذج النهائي موجود أصلاً: {FT}")
    print("لتدريب جديد كلياً: غيّر RUN_NAME في الخلية 2 (أو احذف ft_model.pth ثم أعد هذه الخلية).")
else:
    BATCH    = int(CFG["_BATCH"] or 2)
    MAXS     = int(CFG["STEPS"])
    WORKERS  = int(CFG["TRAIN_WORKERS"])
    SAVE_EVERY = int(CFG["SAVE_EVERY"])
    DRV_RUN  = drive_run_dir() if CFG.get("DRIVE_BACKUP", True) else None
    KEEP     = int(CFG.get("DRIVE_CKPT_KEEP", 2))

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _log_path = RUN_DIR / "train_console.log"

    # ---- مرآة Drive اللحظية: غلاف حول torch.save ينسخ النقاط فور حفظها ----
    # (يُركَّب قبل إنشاء Trainer حتى تغطي كل عمليات الحفظ؛ النسخ متزامن داخل
    #  عملية التدريب نفسها — بلا خيط خلفي ولا عملية إضافية)
    _real_save = torch.save
    def _save_with_drive_mirror(obj, f, *a, **kw):
        _real_save(obj, f, *a, **kw)
        if DRV_RUN and isinstance(f, (str, os.PathLike)):
            p = str(f)
            base = os.path.basename(p)
            if base.startswith("DiT_epoch_") and base.endswith(".pth"):
                try:
                    os.makedirs(DRV_RUN, exist_ok=True)
                    dst = os.path.join(DRV_RUN, base)
                    tmp = dst + ".part"
                    shutil.copy2(p, tmp)
                    os.replace(tmp, dst)
                    # تشذيب النقاط القديمة على Drive (أبقِ آخر KEEP)
                    cks = sorted(glob.glob(os.path.join(DRV_RUN, "DiT_epoch_*_step_*.pth")),
                                 key=lambda x: int(re.search("step_([0-9]+)", x).group(1)))
                    for old in (cks[:-KEEP] if KEEP > 0 else []):
                        try: os.remove(old)
                        except OSError: pass
                    print(f"[drive-mirror] {base} -> Drive", flush=True)
                except Exception as e:
                    print(f"[drive-mirror] فشل نسخ {base}: {e} (التدريب يتابع)", flush=True)
            elif base == "ft_model.pth":
                try:
                    os.makedirs(DRV_RUN, exist_ok=True)
                    shutil.copy2(p, os.path.join(DRV_RUN, base))
                    shutil.copy2(_log_path, os.path.join(DRV_RUN, "train_console.log"))
                    print("[drive-mirror] ft_model.pth + السجل -> Drive (اكتمل التدريب)", flush=True)
                except Exception as e:
                    print(f"[drive-mirror] فشل نسخ النموذج النهائي: {e}", flush=True)
    torch.save = _save_with_drive_mirror

    # ---- سجل مزدوج: الشاشة (خلية الدفتر) + train_console.log ----
    class _Tee(io.TextIOBase):
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

    print(f"[train] بدء الضبط الدقيق: {MAXS} خطوة، دفعة {BATCH}، حفظ كل {SAVE_EVERY}", flush=True)
    print(f"[train] التشغيل داخل نواة الدفتر مباشرة (import train.Trainer) — بلا أي عملية ابن", flush=True)
    if DRV_RUN:
        print(f"[train] مرآة Drive مفعّلة: كل نقطة تُنسخ فور حفظها إلى {DRV_RUN} (أبقِ آخر {KEEP})", flush=True)

    t0 = time.time()
    try:
        with contextlib.redirect_stdout(_Tee(sys.stdout, _log_path)):
            trainer = SEED_TRAIN.Trainer(
                config_path=str(SEEDVC_CFG),       # إعداد seed-vc الرسمي (cwd = seed-vc منذ الخلية 6)
                pretrained_ckpt_path=None,         # يجد آخر نقطة محلية تلقائياً أو ينزّل الأساسية المنشورة
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
        torch.save = _real_save    # أعِد torch.save الأصلي مهما حدث

    mins = (time.time() - t0) / 60
    if FT.exists():
        print(f"\n[اكتمل] التدريب انتهى في {mins:.0f} دقيقة — النموذج النهائي: {FT}")
        if DRV_RUN:
            print(f"[Drive] النقاط والسجل محفوظة في: {DRV_RUN}")
    else:
        tail = ""
        if _log_path.exists():
            tail = "\n".join(_log_path.read_text(errors="replace").splitlines()[-25:])
        raise RuntimeError(
            f"توقف التدريب دون إنتاج النموذج النهائي بعد {mins:.0f} دقيقة.\n"
            f"آخر السجل:\n{tail}\n"
            "أعد هذه الخلية — إن كانت هناك نقطة محفوظة سيُستأنف التدريب منها تلقائياً.")

# منحنى الخسارة من السجل (إن وُجدت نقاط كافية)
STEP_RE = re.compile(r"epoch ([0-9]+), step ([0-9]+), loss: ([0-9.eE+-]+)")
logp = RUN_DIR / "train_console.log"
pts = []
if logp.exists():
    for ln in logp.read_text(errors="replace").splitlines():
        m = STEP_RE.search(ln)
        if m:
            pts.append((int(m.group(2)), float(m.group(3))))
if len(pts) >= 2:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs, ys = zip(*pts)
    fig, ax = plt.subplots(figsize=(9, 3.2), constrained_layout=True)
    ax.plot(xs, ys, lw=1.3, color="#d62728")
    ax.set_xlabel("step")
    ax.set_ylabel("EMA loss")
    ax.set_title(f"loss curve - {CFG['RUN_NAME']} ({len(pts)} points)")
    ax.grid(alpha=0.3)
    chart = RUN_DIR / "loss_curve.png"
    fig.savefig(chart, dpi=110)
    plt.close(fig)
    print(f"منحنى الخسارة: {chart} ({len(pts)} نقطة)")
    # نسخه إلى Drive إن وُجد
    drv = drive_run_dir()
    if drv:
        try:
            shutil.copy2(chart, drv / chart.name)
        except Exception:
            pass

print("\nالخطوة التالية: خلية التحقق (13) — استماع حقيقي بالنقطة الناتجة.")'''))

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

# ============================================================ CODE 13 — verify (in-kernel)
CELLS_C.append(("code", r'''# ============================================================ #
#  الخلية ١٣ — التحقق: استدلال حقيقي بالنقطة المضبوطة          #
#  (مصدر من أمثلة Seed-VC + مرجع عراقي من بياناتك -> تشغيل)    #
#  استدعاء مباشر لـ SEED_INFER.main (المستوردة في الخلية 6)    #
#  داخل نواة الدفتر — بلا عملية خارجية وبلا سطر أوامر      #
# ============================================================ #
from IPython.display import Audio, display
from pathlib import Path
import types

FT = RUN_DIR / "ft_model.pth"
assert FT.exists(), "لا يوجد ft_model.pth — شغّل خلية التدريب (12) حتى الاكتمال"

# اختيار مرجع عراقي من البيانات الجاهزة (مثل منطق cmd_verify في خط البيانات)
ref = None
for d in ("yodas", "youssef", "lahgtna", "kharrufa", "shiekh", "safaa"):
    dd = DATA_DIR / d
    if dd.is_dir():
        cand = sorted(dd.glob("*.ogg"))[:5]
        if cand:
            ref = cand[0]
            break
assert ref is not None, "لا توجد مقاطع عراقية للمرجع — شغّل خلية الأرشيف (7)"
src = SEEDVC / "examples" / "source" / "source_s1.wav"
if not src.exists():
    src = next(iter(sorted((SEEDVC / "examples" / "source").iterdir())))
outdir = RUN_DIR / "verify_out"
outdir.mkdir(parents=True, exist_ok=True)

print(f"المصدر (إنجليزي): {src.name}")
print(f"المرجع (عراقي)  : {ref.relative_to(DATA_DIR)}")
print("تشغيل الاستدلال داخل النواة ... (1-3 دقائق)")

# الأرجس نفسها التي يمررها سطر الأمد لـ inference.py — لكن بلا سطر أوامر:
# نستدعي main() مباشرة على كائن args (SEED_INFER مستوردة من الخلية 6)
vc_args = types.SimpleNamespace(
    source=str(src),
    target=str(ref),
    output=str(outdir),
    diffusion_steps=16, length_adjust=1.0, inference_cfg_rate=0.7,
    f0_condition=False, auto_f0_adjust=False, semi_tone_shift=0,
    checkpoint=str(FT),
    config=str(SEEDVC_CFG),
    fp16=True,
)
try:
    SEED_INFER.main(vc_args)     # الاستدلال كاملاً داخل نواة الدفتر
    verify_ok = True
except Exception as e:
    print(f"[تحقق] فشل الاستدلال: {e}")
    verify_ok = False

outs = sorted(outdir.glob("vc_*.wav"))
if outs:
    print(f"\nناتج التحويل: {outs[0].name}  ({outs[0].stat().st_size/1e6:.1f} MB)")
    print("استمع للناتج (المصدر إنجليزي يحمل الطابع الصوتي العراقي المرجعي):")
    display(Audio(filename=str(outs[0])))
    for extra in outs[1:3]:
        print(f"ناتج إضافي: {extra.name}")
        display(Audio(filename=str(extra)))
else:
    print("لم يُنتج ملف صوتي — راجع السجل أعلاه. (غير قاتل: الأوزان ستصدَّر في الخلية 14 على أي حال)")'''))

# ============================================================ CODE 14 — export (direct call)
CELLS_C.append(("code", r'''# ============================================================ #
#  الخلية ١٤ — تصدير الحزمة النهائية                           #
#  أوزان + إعداد + تقرير + سجلات — محلياً ثم نسخها إلى Drive   #
#  استدعاء مباشر لـ PIPELINE.cmd_export داخل نواة الدفتر       #
#  (نفس الواجهة، بلا عملية خارجية وبلا سطر أوامر)            #
# ============================================================ #
import shutil, types, json
from pathlib import Path

FT = RUN_DIR / "ft_model.pth"
assert FT.exists(), "لا يوجد ft_model.pth — أكمل التدريب أولاً (الخلية 12)"
EXPORT_DIR = EXPORTS / CFG["RUN_NAME"]

# استدعاء cmd_export من وحدة خط البيانات المستوردة (الخلية 6) — داخل النواة
PIPELINE.cmd_export(types.SimpleNamespace(
    run_dir=str(RUN_DIR),
    config=str(SEEDVC_CFG),
    data_dir=str(DATA_DIR),
    out=str(EXPORT_DIR),
))
assert (EXPORT_DIR / "seed_uvit_tat_xlsr_tiny_iraqi.pth").exists(), "فشل التصدير — راجع السجل"

# توثيق إضافي يُنسخ مع الحزمة (قابلية إعادة الإنتاج جزء من الجودة)
for extra in (RUN_DIR / "train_console.log", RUN_DIR / "loss_curve.png",
              RUN_DIR / "config_dit_mel_seed_uvit_xlsr_tiny.yml", DATA_DIR / "_stats.json"):
    if extra.exists():
        shutil.copy2(extra, EXPORT_DIR / extra.name)
        print(f"  + {extra.name}")

total = 0
print(f"\\nحزمة التصدير (محلي): {EXPORT_DIR}")
for p in sorted(EXPORT_DIR.rglob("*")):
    if p.is_file():
        s = p.stat().st_size
        total += s
        print(f"  {p.name:<46} {s/1e6:9.2f} MB")
print(f"  {'الإجمالي':<46} {total/1e6:9.2f} MB")

# ---- النسخ الفوري إلى Drive (ديمومة فورية) ----
DRV_EXP = drive_export_base()
if DRV_EXP:
    print(f"\\nنسخ الحزمة إلى Drive: {DRV_EXP / CFG['RUN_NAME']} ...")
    (DRV_EXP / CFG["RUN_NAME"]).mkdir(parents=True, exist_ok=True)
    for p in sorted(EXPORT_DIR.rglob("*")):
        if p.is_file():
            shutil.copy2(p, DRV_EXP / CFG["RUN_NAME"] / p.name)
    print("[Drive] الحزمة محفوظة بشكل دائم — تنجو من أي انقطاع جلسة.")'''))

# ============================================================ CODE 15 — archive
CELLS_C.append(("code", r'''# ============================================================ #
#  الخلية ١٥ — أرشيف tar.gz واحد (في Drive + محلي)             #
# ============================================================ #
import shutil
from pathlib import Path

EXPORT_DIR = EXPORTS / CFG["RUN_NAME"]
assert (EXPORT_DIR / "seed_uvit_tat_xlsr_tiny_iraqi.pth").exists(), "شغّل خلية التصدير (14) أولاً"

base = EXPORTS / f"rafidain_seedvc_{CFG['RUN_NAME']}"
archive = shutil.make_archive(str(base), "gztar", root_dir=str(EXPORTS), base_dir=EXPORT_DIR.name)
size_mb = Path(archive).stat().st_size / 1e6
print(f"الأرشيف جاهز: {archive}  ({size_mb:.1f} MB)")

# ---- نسخه إلى Drive ----
if globals().get("DRIVE_OK") and globals().get("DRIVE_ROOT"):
    drv_arc = globals()["DRIVE_ROOT"] / "exports" / Path(archive).name
    shutil.copy2(archive, drv_arc)
    print(f"[Drive] نسخة دائمة: {drv_arc}")

print(R"""
لتنزيله إلى جهازك:
  • من Drive (الأسهل): drive.google.com <- مجلد rafidain_seedvc/exports <- نزّل الملف
  • أو من لوحة ملفات Colab (الشريط الجانبي): مجلد exports <- كليك يمين <- Download
  • أو أضف في خلية جديدة:  from google.colab import files; files.download("<مسار الأرشيف>")
""")
'''))

# ============================================================ MD 7 — persistence layers
CELLS_C.append(("markdown", r'''<div dir="rtl">

## الحفظ الدائم — ثلاث طبقات (حتى تنزّلها بعد سنة)

| الطبقة | أين تبقى | كم تدوم | كيف تنزّلها لاحقاً |
|---|---|---|---|
| 1. **Google Drive** | `MyDrive/rafidain_seedvc/` — نقاط أثناء التدريب (من مرآة الخلية 12) + حزمة التصدير + الأرشيف + نسخة أرشيف البيانات الجاهز | دائمة في حسابك مهما انقطعت جلسات Colab | من أي متصفح/جهاز عبر drive.google.com |
| 2. **HuggingFace Hub** | مستودع في حسابك (خاص افتراضياً) | **دائمة مجاناً — بلا علاقة بـ Google إطلاقاً** | `hf download <repo>` أو زر التنزيل من صفحة المستودع |
| 3. القرص المحلي `/content` | البيانات الجاهزة + البيئة + النقاط | تضيع عند انقطاع/إعادة تعيين الجلسة | (قابلة لإعادة البناء بالخلايا 5-7 — كلها استئنافية) |

> **الوصية**: الطبقة 2 هي ضمانتك الحقيقية للمدى البعيد — حتى لو فقدت حساب Google أو حُذف Drive،
> يبقى النموذج في HuggingFace قابلاً للتنزيل من أي مكان في العالم. تحتاج فقط رمز API بصلاحية كتابة
> من `hf.co/settings/tokens` تضعه في الخلية 2 ثم تشغّل الخلية 16.

ما يصل إلى Drive **أثناء التدريب** (مرآة الخلية 12، لحظياً عند كل حفظ): آخر نقطتين
`DiT_epoch_*.pth` + `ft_model.pth` عند الاكتمال + `train_console.log` + `loss_curve.png` —
كل ذلك في `MyDrive/rafidain_seedvc/runs/<اسم-التشغيل>/`، بمساحة لا تتجاوز ~2GB.

</div>'''))

# ============================================================ CODE 16 — HF Hub upload
CELLS_C.append(("code", r'''# ============================================================ #
#  الخلية ١٦ — الرفع الدائم إلى HuggingFace Hub                #
#  النموذج يُخزَّن في حسابك أنت، مجاناً، بلا علاقة بـ Google     #
#  (أعد تشغيلها في أي وقت مستقبلاً — ولو بعد شهور)              #
#  عبر huggingface_hub مباشرة — بلا سكربت مؤقت ولا سطر أوامر   #
# ============================================================ #
from pathlib import Path

EXPORT_DIR = EXPORTS / CFG["RUN_NAME"]

if not CFG["HF_TOKEN"]:
    print("لا يوجد رمز HuggingFace بعد — لا مشكلة: كل أصولك محفوظة على Drive،")
    print("ويمكنك تنفيذ الرفع الدائم في أي وقت لاحق (الخلية تعمل مستقلة):\\n")
    print("  1) أنشئ حساباً مجانياً في huggingface.co إن لم يكن لديك")
    print("  2) من hf.co/settings/tokens أنشئ New Token بصلاحية Write وانسخه")
    print('  3) في الخلية 2 اضبط:  CFG["HF_TOKEN"] = "hf_ضع-الرمز-هنا"  ثم أعد تشغيلها')
    print("  4) أعد تشغيل هذه الخلية — سترفع الحزمة كاملة إلى مستودع خاص في حسابك")
else:
    assert (EXPORT_DIR / "seed_uvit_tat_xlsr_tiny_iraqi.pth").exists(), "شغّل خلية التصدير (14) أولاً"
    # الرفع مباشرة داخل النواة — huggingface_hub مثبتة في الخلية 5 (بلا عملية خارجية)
    import time
    from huggingface_hub import HfApi
    api = HfApi(token=str(CFG["HF_TOKEN"]))
    user = api.whoami()["name"]
    repo_id = str(CFG.get("HF_REPO_ID", "")).strip() or f"{user}/rafidain-seedvc-iraqi"
    private = bool(CFG["HF_PRIVATE"])
    api.create_repo(repo_id, private=private, exist_ok=True)
    print(f"[hf] رفع {EXPORT_DIR} -> {repo_id} (خاص = {private}) ...")
    api.upload_folder(
        folder_path=str(EXPORT_DIR),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Rafidain Iraqi fine-tune ({time.strftime('%Y-%m-%d %H:%M')})",
    )
    print(f"[hf] REPO_URL: https://huggingface.co/{repo_id}")
    print(R"""
[تم] النموذج الآن في حسابك على HuggingFace — بشكل دائم.
لتنزيله في أي وقت مستقبلاً من أي جهاز (بلا Google ولا Colab):
    pip install -U "huggingface_hub[cli]"
    hf download <اسم-حسابك>/rafidain-seedvc-iraqi --local-dir ./mymodel
أو من بايثون:
    from huggingface_hub import snapshot_download
    snapshot_download("<اسم-حسابك>/rafidain-seedvc-iraqi", local_dir="./mymodel")
أو مباشرة من صفحة المستودع في المتصفح.""")
'''))

# ============================================================ CODE 17 — cleanup
CELLS_C.append(("code", r'''# ============================================================ #
#  الخلية ١٧ — (اختيارية) تحرير مساحة القرص المحلي            #
#  تحذف البيانات القابلة لإعادة الجلب فقط — وتبقى النقاط       #
#  والحزم وكل ما على Drive كما هو                             #
#  (بيانات الأرشيف الجاهز تُعاد في دقائق من نسخة Drive)         #
# ============================================================ #
import shutil
from pathlib import Path

def dir_gb(p: Path) -> float:
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9

targets = [DATA_DIR, ROOT / "raw", SEEDVC / "checkpoints" / "hf_cache"]
labels = ["بيانات التدريب الجاهزة (الأرشيف)", "مصادر خام إضافية", "مخبأ تنزيل HF"]
print(f"المساحة الحرة الآن: {shutil.disk_usage(ROOT).free/1e9:.1f} GB")
for t, lb in zip(targets, labels):
    s = dir_gb(t)
    print(f"  {t.name:<14} {s:6.2f} GB   ({lb})")

if CFG["CLEAN_AFTER"]:
    for t in targets[:2]:   # البيانات والخام — أما مخبأ HF فيُحذف فقط لو الطلب صريح
        if t.exists():
            shutil.rmtree(t)
            print(f"  حُذف: {t}")
    if (ROOT / ".ready_archive_ok").exists():
        (ROOT / ".ready_archive_ok").unlink()
    import sys as _sys
    sh([_sys.executable, "-m", "pip", "cache", "purge"], check=False)
    print(f"المساحة الحرة بعد التنظيف: {shutil.disk_usage(ROOT).free/1e9:.1f} GB")
    print("كل ما يهم محفوظ على Drive (النقاط + الحزمة + الأرشيف) — لا خسارة دائمة.")
else:
    print(R"""
للتفعيل: اجعل  CFG["CLEAN_AFTER"] = True  في الخلية 2 ثم أعد تشغيل هذه الخلية.""")'''))

# ============================================================ CODE 18 — deployment
CELLS_C.append(("code", r'''# ============================================================ #
#  الخلية ١٨ — النشر: كيف تستخدم النموذج المضبوط؟              #
#  (خلية إرشادية تطبع الأوامر الجاهزة بمساراتك الفعلية)        #
# ============================================================ #
from pathlib import Path

EXPORT_DIR = EXPORTS / CFG["RUN_NAME"]
W = EXPORT_DIR / "seed_uvit_tat_xlsr_tiny_iraqi.pth"
C = EXPORT_DIR / "config_dit_mel_seed_uvit_xlsr_tiny.yml"

if not W.exists():
    print("شغّل خلية التصدير (14) أولاً حتى تتوفر الحزمة.")
else:
    print(f"""
==========================================================
 نشر النموذج المضبوط — ثلاثة مسارات جاهزة
==========================================================

(أ) استدلال فوري في هذه الجلسة — استيراد مباشر داخل النواة
    (الوحدات محمّلة أصلاً منذ الخلية 6 — بلا أي عملية خارجية):
    في خلية جديدة الصق:

      import types
      vc = types.SimpleNamespace(
          source=r"<صوت-المتكلم-الأصلي.wav>",
          target=r"<صوت-مرجعي-عراقي.wav>",   # مقطع عراقي من بياناتك
          output=r"./out",
          diffusion_steps=30, length_adjust=1.0, inference_cfg_rate=0.7,
          f0_condition=False, auto_f0_adjust=False, semi_tone_shift=0,
          checkpoint=r"{W}", config=r"{C}", fp16=True)
      SEED_INFER.main(vc)     # الاستدلال داخل النواة مباشرة

(ب) خدمة رافِدَين الكاملة على نفس جلسة Colab — REST + WebSocket + واجهة ويب
    بنفق عام مجاني (trycloudflare) للاستنساخ اللحظي من أي مكان.
    المستودع منسوخ هنا: {RAFIDAIN}
    في خلية جديدة الصق ما يلي (توجيه الخدمة إلى أوزانك المضبوطة):

      import os
      os.environ["RAFIDAIN__ENGINE__LOCAL_CHECKPOINT"] = r"{W}"
      os.environ["RAFIDAIN__ENGINE__LOCAL_CONFIG"]     = r"{C}"
      os.environ["RAFIDAIN__ENGINE__DEVICE"]           = "cuda"
      os.environ["RAFIDAIN__API_KEYS"]                 = r'["<مفتاح-API-عشوائي-خاص-بك>"]'
      # تشغيل سكربت تثبيت خدمة رافِدَين الرسمي (أمر نظام لتشغيل الخدمة):
      !bash {RAFIDAIN}/scripts/install/setup_colab.sh

    بعد دقائق يطبع رابطاً عاماً مؤقتاً:
      لوحة التحكم:  <الرابط>/web   |   البث الحي:  <الرابط>/live   |   API:  <الرابط>/docs
    (متغيرا البيئة LOCAL_CHECKPOINT / LOCAL_CONFIG هما المساران أعلاه — يُقرآن مباشرة من البيئة)

(ج) على أي جهاز GPU آخر (خادمك / RunPod / غيره):
    1) نزّل الحزمة من Drive (rafidain_seedvc/exports) أو من HuggingFace (الخلية 16)
    2) git clone https://github.com/Plachtaa/seed-vc && cd seed-vc
    3) python inference.py --checkpoint <الحزمة>/seed_uvit_tat_xlsr_tiny_iraqi.pth \\
           --config <الحزمة>/config_dit_mel_seed_uvit_xlsr_tiny.yml \\
           --source a.wav --target ref.wav --output ./out
    أو خدمة رافِدَين الكاملة بنفس متغيرَي البيئة في (ب) — الدليل: DEPLOYMENT.md و WEBUI.md

تذكير أخلاقي: استنساخ صوت شخص حقيقي يتطلب موافقته الصريحة الموثقة،
وأفصح للطرف الآخر عند ما يوجبه القانون — راجع قسم الأمان في README المستودع.
""")'''))

# ============================================================ MD 8 — troubleshooting
CELLS_C.append(("markdown", r'''<div dir="rtl">

## حل المشكلات الشائعة

| المشكلة | السبب | الحل |
|---|---|---|
| `Runtime disconnected` قديماً | النسخة الأولى كانت تشغّل التدريب كعملية خلفية + مراقب دائم + حلقة انتظار — أنماط «استخدام غير مسموح» في الطبقة المجانية | **أُزيلت كلها في هذه النسخة**: التدريب داخل نواة الخلية 12 مباشرة — لا عمليات خفية ولا عملية ابن إطلاقاً |
| انقطعت الجلسة أثناء التدريب | حد طبيعي لجلسات Colab | `Run all` — البيئة تُبنى، الأرشيف يُفك من Drive، والخلية 11 تستعيد آخر نقطة، والتدريب يكمل تلقائياً |
| انقطع تنزيل الأرشيف في المنتصف | شبكة | أعد الخلية 7 — `curl -C -` يستكمل من حيث توقف (تحقق الحجم الدقيق بالبايت قبل أي فك) |
| نفدت مساحة القرص | ت accumulation البيانات | خلية التنظيف (17) — أو قلّل مساحة بنسخ أرشيف Drive فقط عند الحاجة |
| `CUDA out of memory` أثناء التدريب | دفعة أكبر من الذاكرة | الخلية 2: `CFG["BATCH"] = 1` ثم أعد الخليتين 3 و12 |
| فشل تنزيل نقطة HF | شبكة/قيود IP | أعد الخلية 10 — التنزيل قابل للاستئناف؛ أو اضبط `HF_ENDPOINT` مرآة في الخلية 2 |
| التدريب بطيء جداً | T4 + مقاطع طويلة | طبيعي على الطبقة المجانية؛ أو قلّل `STEPS`؛ L4 يضاعف السرعة |
| فشل `verify` | مقطع مرجعي شاذ | غير قاتل — الأوزان تُصدَّر على أي حال (الخلية 14) |

**بعد أي انقطاع**: افتح الدفتر ← `Run all` ← كل شيء يستأنف: الحزم (دقائق)،
البيانات (من Drive: دقائق / من الإنترنت: حتى 20 دقيقة)، النقاط (ثوانٍ)، والتدريب
يكمل من آخر خطوة محفوظة. لا يوجد سيناريو يبدأ فيه الدفتر من الصفر إلا مرة واحدة.

</div>'''))

# ============================================================ MD 9 — final summary
CELLS_C.append(("markdown", r'''<div dir="rtl">

## الخلاصة — ماذا حصلت عليه؟

1. **نموذج مضبوط** `seed_uvit_tat_xlsr_tiny_iraqi.pth` — Seed-VC tiny (25M باراميتر)
   تكيّف على الساعات العراقية الجاهزة، عبر تدريب داخل نواة الدفتر مباشرة (import train ← Trainer) بلا أي عملية خلفية أو عملية ابن.
2. **ثلاث نسخ دائمة**: Drive (نقاط + حزمة + أرشيف)، tar.gz واحد قابل للتنزيل،
   وHuggingFace Hub (اختياري لكنه الأضمن).
3. **قابلية استئناف كاملة**: كل خلية تتخطى ما أنجز، والأرشيف الجاهز جعل مرحلة
   البيانات دقائق بدل ساعات، والتدريب يستأنف من آخر نقطة.
4. **مسارات نشر جاهزة**: استدلال فوري (استيراد inference)، خدمة رافِدَين الكاملة
   (REST/WebSocket/ويب) على نفس الجلسة، أو أي جهاز GPU آخر.

**الخطوة التالية المقترحة**: شغّل التحقق (13) واستمع للناتج — ثم إن أعجبك، ارفع
النقطة إلى HuggingFace (16) وجرّب خدمة البث الحي (18-ب) بصوتك أنت.

</div>'''))
