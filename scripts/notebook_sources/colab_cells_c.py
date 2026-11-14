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

## مرحلة التدريب — في مقدمة الخلية، بلا عمليات خلفية

- **نقطة البداية دائماً** الأوزان المنشورة `DiT_uvit_tat_xlsr_ema.pth` من `Plachta/Seed-VC`
  (لا تدريب من الصفر أبداً). إن وُجدت نقطة سابقة في مجلد التشغيل يُستأنف منها تلقائياً
  (سلوك المدرب الرسمي: يبحث عن أحدث `DiT_epoch_*_step_*.pth` ويكمل منها، ويُبقي آخر
  نقطتين فقط).
- **كيف يعمل هنا؟** نستورد وحدة `train.py` من مستودع Seed-VC المستنسخ وننشئ `Trainer`
  مباشرة ثم نستدعي `.train()` — التدريب يعمل **داخل خلية النوت بوك نفسها** (في البيئة
  المعزولة)، فترى شريط تقدم tqdm وأسطر `epoch E, step S, loss: L` تظهر مباشرة أمامك.
  هذا هو النمط الذي تسمحه سياسة Colab المجانية: استخدام تفاعلي حقيقي للنواة.
- **لماذا لا عملية خلفية؟** النسخة الأولى كانت تطلق التدريب كعملية مستقلة تعيش بعد
  انتهاء الخلية، مع مراقب نسخ دائم وحلقة انتظار لساعات — وهذا يُصنَّف «استخداماً
  غير مسموح» في الطبقة المجانية فتُقطع الجلسة فجأة (Runtime disconnected).
  هنا لا يوجد أي من ذلك.
- **النسخ إلى Drive رغم ذلك — كيف؟** نغلّف `torch.save` (التي يستدعيها المدرب عند
  كل نقطة حفظ) بغلاف ينسخ النقطة الجديدة إلى Drive **فور حفظها من داخل عملية
  التدريب نفسها** — بلا مراقب ولا خيط خلفي. أقصى خسارة عند أي انقطاع = الخطوات منذ
  آخر `SAVE_EVERY` (250 افتراضياً ≈ دقائق معدودة على T4).
- **مخرجات التدريب**: سطر خسارة كل 10 خطوات، نقطة كل 250 خطوة (تُنسخ فوراً)، وفي
  النهاية `ft_model.pth` (أوزان DiT المضبوطة — كل ما يلزم للنشر والاستدلال) ثم تُنسخ
  هي أيضاً إلى Drive مع السجل الكامل.
- **الإعدادات الموصى بها من دليل المستودع (FINETUNING.md)**: دفعة 2 لذاكرة 16GB (T4)
  و4 لـ 24GB (L4)؛ 200–500 خطوة لمتحدث واحد؛ **1500–4000 خطوة لتكيّف عام مع اللهجة**
  (3000 افتراضياً هنا)؛ معدل التعلم مثبّت في الإعداد الرسمي (1e-5) — لا ترفعه لنموذج
  25M باراميتر.
- **انقطعت الجلسة؟** افتح الدفتر ← **Run all**: تُبنى البيئة، والأرشيف الجاهز يُفك من
  نسخة Drive إن وُجدت، والخلية 11 تستعيد أحدث نقطة من Drive، ويستأنف التدريب من
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

# ============================================================ CODE 12 — foreground training
CELLS_C.append(("code", r'''# ============================================================ #
#  الخلية ١٢ — التدريب: في مقدمة هذه الخلية مباشرة             #
#  استيراد train.py من seed-vc وإنشاء Trainer وتشغيله —        #
#  بلا عملية خلفية ولا PID ولا مراقب ولا حلقة انتظار          #
#  كل نقطة حفظ تُنسخ فوراً إلى Drive من داخل عملية التدريب     #
#  (غلاف torch.save) — والسجل الكامل يُحفظ في train_console.log#
# ============================================================ #
import os, sys, time, json, subprocess, contextlib
from pathlib import Path

FT = RUN_DIR / "ft_model.pth"

if FT.exists():
    print(f"النموذج النهائي موجود أصلاً: {FT}")
    print("لتدريب جديد كلياً: غيّر RUN_NAME في الخلية 2 (أو احذف ft_model.pth ثم أعد هذه الخلية).")
else:
    BATCH  = int(CFG["_BATCH"] or 2)
    MAXS   = int(CFG["STEPS"])
    WANTS  = int(CFG["TRAIN_WORKERS"])
    DRV_RUN = drive_run_dir() if CFG.get("DRIVE_BACKUP", True) else None
    KEEP   = int(CFG.get("DRIVE_CKPT_KEEP", 2))

    # برنامج التدريب: يستورد train.py من seed-vc، يغلّف torch.save بمرآة Drive،
    # ويعيد التوجيه المزدوج (stdout) إلى الشاشة و train_console.log معاً
    # (TQ = فاصل ثلاثي-اقتباس يُبنى بـ chr(34) حتى لا يتصادم مع غلاف الخلية)
    TQ = chr(34) * 3
    TRAIN_DRIVER = (
        "import os, sys, time, json, shutil, contextlib, io\n"
        "import torch\n"
        "\n"
        "SEEDVC   = sys.argv[1]\n"
        "RUN_DIR  = sys.argv[2]\n"
        "DATA_DIR = sys.argv[3]\n"
        "CFG_PATH = sys.argv[4]\n"
        "DRV_RUN  = sys.argv[5] or None\n"
        "KEEP     = int(sys.argv[6])\n"
        "BATCH    = int(sys.argv[7])\n"
        "MAXS     = int(sys.argv[8])\n"
        "SAVE_EVERY = int(sys.argv[9])\n"
        "WORKERS  = int(sys.argv[10])\n"
        "\n"
        "CFG = json.load(open(CFG_PATH, encoding=\"utf-8\"))\n"
        "\n"
        "os.chdir(SEEDVC)                    # train.py يفتح configs/hifigan.yml و./checkpoints نسبياً\n"
        "sys.path.insert(0, SEEDVC)\n"
        "sys.path.insert(0, os.path.join(SEEDVC, \"modules\"))\n"
        "\n"
        "import train as seed_train          # وحدة المدرب الرسمية — بلا سطر أوامر\n"
        "\n"
        "class Tee(io.TextIOBase):\n"
        "    \"\"\"stdout مزدوج: الشاشة (تُمرر عبر الأنبوب للنواة) + ملف السجل.\"\"\"\n"
        "    def __init__(self, original, path):\n"
        "        self.original = original\n"
        "        self.file = open(path, \"a\", encoding=\"utf-8\", buffering=1)\n"
        "    def write(self, s):\n"
        "        self.original.write(s)\n"
        "        self.file.write(s)\n"
        "        return len(s)\n"
        "    def flush(self):\n"
        "        self.original.flush()\n"
        "        self.file.flush()\n"
        "\n"
        "_log_path = os.path.join(RUN_DIR, \"train_console.log\")\n"
        "os.makedirs(RUN_DIR, exist_ok=True)\n"
        "\n"
        "# ---- مرآة Drive اللحظية: غلاف حول torch.save ينسخ النقاط فور حفظها ----\n"
        "_real_save = torch.save\n"
        "def _save_with_drive_mirror(obj, f, *a, **kw):\n"
        "    _real_save(obj, f, *a, **kw)\n"
        "    if DRV_RUN and isinstance(f, (str, os.PathLike)):\n"
        "        p = str(f)\n"
        "        base = os.path.basename(p)\n"
        "        if base.startswith(\"DiT_epoch_\") and base.endswith(\".pth\"):\n"
        "            try:\n"
        "                os.makedirs(DRV_RUN, exist_ok=True)\n"
        "                dst = os.path.join(DRV_RUN, base)\n"
        "                tmp = dst + \".part\"\n"
        "                shutil.copy2(p, tmp)\n"
        "                os.replace(tmp, dst)\n"
        "                # تشذيب النقاط القديمة على Drive (أبقِ آخر KEEP)\n"
        "                import glob as _g, re as _re\n"
        "                cks = sorted(_g.glob(os.path.join(DRV_RUN, \"DiT_epoch_*_step_*.pth\")),\n"
        "                             key=lambda x: int(_re.search(\"step_([0-9]+)\", x).group(1)))\n"
        "                for old in (cks[:-KEEP] if KEEP > 0 else []):\n"
        "                    try: os.remove(old)\n"
        "                    except OSError: pass\n"
        "                print(f\"[drive-mirror] {base} -> Drive\", flush=True)\n"
        "            except Exception as e:\n"
        "                print(f\"[drive-mirror] فشل نسخ {base}: {e} (التدريب يتابع)\", flush=True)\n"
        "        elif base == \"ft_model.pth\":\n"
        "            try:\n"
        "                os.makedirs(DRV_RUN, exist_ok=True)\n"
        "                shutil.copy2(p, os.path.join(DRV_RUN, base))\n"
        "                shutil.copy2(_log_path, os.path.join(DRV_RUN, \"train_console.log\"))\n"
        "                print(\"[drive-mirror] ft_model.pth + السجل -> Drive (اكتمل التدريب)\", flush=True)\n"
        "            except Exception as e:\n"
        "                print(f\"[drive-mirror] فشل نسخ النموذج النهائي: {e}\", flush=True)\n"
        "torch.save = _save_with_drive_mirror\n"
        "\n"
        "t0 = time.time()\n"
        "print(f\"[train] بدء الضبط الدقيق: {MAXS} خطوة، دفعة {BATCH}، حفظ كل {SAVE_EVERY}\", flush=True)\n"
        "if DRV_RUN:\n"
        "    print(f\"[train] مرآة Drive مفعّلة: كل نقطة تُنسخ فور حفظها إلى {DRV_RUN} (أبقِ آخر {KEEP})\", flush=True)\n"
        "\n"
        "with contextlib.redirect_stdout(Tee(sys.stdout, _log_path)):\n"
        "    trainer = seed_train.Trainer(\n"
        "        config_path=os.path.join(SEEDVC, \"configs\", \"presets\", \"config_dit_mel_seed_uvit_xlsr_tiny.yml\"),\n"
        "        pretrained_ckpt_path=None,      # يجد آخر نقطة محلية تلقائياً أو ينزّل الأساسية المنشورة\n"
        "        data_dir=DATA_DIR,\n"
        "        run_name=CFG[\"RUN_NAME\"],\n"
        "        batch_size=BATCH,\n"
        "        num_workers=WORKERS,\n"
        "        steps=MAXS,\n"
        "        save_interval=SAVE_EVERY,\n"
        "        max_epochs=1000,\n"
        "        device=\"cuda:0\",\n"
        "    )\n"
        "    trainer.train()\n"
        "\n"
        "print(f\"[train] انتهى في {(time.time()-t0)/60:.1f} دقيقة — ft_model.pth جاهز\", flush=True)\n"
    )
    ROOT_TMP = ROOT / "train_driver.py"
    ROOT_TMP.write_text(TRAIN_DRIVER, encoding="utf-8")

    cmd = [str(VENV_PY), "-u", str(ROOT_TMP),
           str(SEEDVC), str(RUN_DIR), str(DATA_DIR), str(CFG_PATH),
           str(DRV_RUN or ""), str(KEEP), str(BATCH), str(MAXS),
           str(int(CFG["SAVE_EVERY"])), str(WANTS)]
    print("أمر التدريب (يعمل الآن في مقدمة هذه الخلية — سترى التقدم مباشرة):")
    print("  " + " ".join(cmd[:2]) + " train_driver.py ...\\n")
    t0 = time.time()
    # ملاحظة سياسة Colab: عملية أمامية تابعة للخلية، بلا فصل عن النواة ولا تخزيد
    # تظهر مخرجاتها هنا مباشرة وتنتهي بانتهاء التدريب — هذا هو النمط المسموح.
    rc = sh(cmd, check=False)
    mins = (time.time() - t0) / 60

    if FT.exists():
        print(f"\\n[اكتمل] التدريب انتهى في {mins:.0f} دقيقة — النموذج النهائي: {FT}")
        if DRV_RUN:
            print(f"[Drive] النقاط والسجل محفوظة في: {DRV_RUN}")
    else:
        tail = ""
        logp = RUN_DIR / "train_console.log"
        if logp.exists():
            tail = "\\n".join(logp.read_text(errors="replace").splitlines()[-25:])
        raise RuntimeError(
            f"توقفت عملية التدريب دون إنتاج النموذج النهائي (رمز {rc}) بعد {mins:.0f} دقيقة.\\n"
            f"آخر السجل:\\n{tail}\\n"
            "أعد هذه الخلية — إن كانت هناك نقطة محفوظة سيُستأنف التدريب منها تلقائياً.")

# منحنى الخسارة من السجل (إن وُجدت نقاط كافية)
import re
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
            import shutil as _sh
            _sh.copy2(chart, drv / chart.name)
        except Exception:
            pass

print("\\nالخطوة التالية: خلية التحقق (13) — استماع حقيقي بالنقطة الناتجة.")'''))

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

# ============================================================ CODE 13 — verify
CELLS_C.append(("code", r'''# ============================================================ #
#  الخلية ١٣ — التحقق: استدلال حقيقي بالنقطة المضبوطة          #
#  (مصدر من أمثلة Seed-VC + مرجع عراقي من بياناتك -> تشغيل)    #
#  عبر استيراد inference.py من seed-vc داخل البيئة المعزولة    #
#  (بلا سطر أوامر للسكربت — نستدعي main() بالأرجس نفسها)       #
# ============================================================ #
from IPython.display import Audio, display
from pathlib import Path
import subprocess, os

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
print("تشغيل الاستدلال ... (1-3 دقائق)")

# استيراد inference.py كوحدة واستدعاء main() بالأرجس نفسها — داخل venv
Q = chr(39) * 3   # فاصل ثلاثي-اقتباس آمن داخل غلاف الخلية
code = (
    "import sys, types, argparse;"
    f"sys.path.insert(0, {Q}{SEEDVC}{Q});"
    f"import os; os.chdir({Q}{SEEDVC}{Q});"
    "import inference as inf;"
    "a = types.SimpleNamespace("
    f"  source={Q}{src}{Q},"
    f"  target={Q}{ref}{Q},"
    f"  output={Q}{outdir}{Q},"
    "  diffusion_steps=16, length_adjust=1.0, inference_cfg_rate=0.7,"
    "  f0_condition=False, auto_f0_adjust=False, semi_tone_shift=0,"
    f"  checkpoint={Q}{FT}{Q},"
    f"  config={Q}{SEEDVC_CFG}{Q},"
    "  fp16=True);"
    "inf.main(a)"
)
r = subprocess.run([str(VENV_PY), "-c", code], env=ENV, text=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(r.stdout[-4000:], flush=True)   # آخر 4KB من السجل (تحميل النماذج مطوّل)

outs = sorted(outdir.glob("vc_*.wav"))
if outs:
    print(f"\\nناتج التحويل: {outs[0].name}  ({outs[0].stat().st_size/1e6:.1f} MB)")
    print("استمع للناتج (المصدر إنجليزي يحمل الطابع الصوتي العراقي المرجعي):")
    display(Audio(filename=str(outs[0])))
    for extra in outs[1:3]:
        print(f"ناتج إضافي: {extra.name}")
        display(Audio(filename=str(extra)))
else:
    print("لم يُنتج ملف صوتي — راجع السجل أعلاه. (غير قاتل: الأوزان ستصدَّر في الخلية 14 على أي حال)")'''))

# ============================================================ CODE 14 — export
CELLS_C.append(("code", r'''# ============================================================ #
#  الخلية ١٤ — تصدير الحزمة النهائية                           #
#  أوزان + إعداد + تقرير + سجلات — محلياً ثم نسخها إلى Drive   #
#  عبر استيراد iraqi_pipeline داخل البيئة المعزولة (cmd_export)#
# ============================================================ #
import shutil, subprocess, json
from pathlib import Path

FT = RUN_DIR / "ft_model.pth"
assert FT.exists(), "لا يوجد ft_model.pth — أكمل التدريب أولاً (الخلية 12)"
EXPORT_DIR = EXPORTS / CFG["RUN_NAME"]

# استدعاء cmd_export من وحدة خط البيانات (نفس الواجهة، بلا سطر أوامر)
Q = chr(39) * 3   # فاصل ثلاثي-اقتباس آمن داخل غلاف الخلية
code = (
    "import types, importlib.util as ilu;"
    f"spec = ilu.spec_from_file_location('iraqi_pipeline', {Q}{PIPE_SRC}{Q});"
    "m = ilu.module_from_spec(spec); spec.loader.exec_module(m);"
    "m.cmd_export(types.SimpleNamespace("
    f"  run_dir={Q}{RUN_DIR}{Q},"
    f"  config={Q}{SEEDVC_CFG}{Q},"
    f"  data_dir={Q}{DATA_DIR}{Q},"
    f"  out={Q}{EXPORT_DIR}{Q}))"
)
r = subprocess.run([str(VENV_PY), "-c", code], env=ENV, text=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(r.stdout, flush=True)
assert r.returncode == 0 and (EXPORT_DIR / "seed_uvit_tat_xlsr_tiny_iraqi.pth").exists(), "فشل التصدير — راجع السجل"

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
    # الرفع من البيئة المعزولة (حيث huggingface_hub مثبت)
    import subprocess, time
    code = (
        "import os, time\n"
        "from huggingface_hub import HfApi\n"
        "token = os.environ['HF_TOKEN']\n"
        "api = HfApi(token=token)\n"
        "user = api.whoami()['name']\n"
        "repo_id = os.environ.get('NB_REPO_ID', '').strip() or user + '/rafidain-seedvc-iraqi'\n"
        "private = os.environ.get('NB_PRIVATE', '1') == '1'\n"
        "api.create_repo(repo_id, private=private, exist_ok=True)\n"
        "print('[hf] upload ' + os.environ['NB_EXPORT_DIR'] + ' -> ' + repo_id + ' (private = ' + str(private) + ')')\n"
        "api.upload_folder(\n"
        "    folder_path=os.environ['NB_EXPORT_DIR'],\n"
        "    repo_id=repo_id,\n"
        "    repo_type='model',\n"
        "    commit_message='Rafidain Iraqi fine-tune (' + time.strftime('%Y-%m-%d %H:%M') + ')')\n"
        "print('[hf] REPO_URL: https://huggingface.co/' + repo_id)\n"
    )
    env = dict(ENV)
    env["HF_TOKEN"] = str(CFG["HF_TOKEN"])
    env["NB_EXPORT_DIR"] = str(EXPORT_DIR)
    env["NB_REPO_ID"] = str(CFG.get("HF_REPO_ID", ""))
    env["NB_PRIVATE"] = "1" if CFG["HF_PRIVATE"] else "0"
    r = subprocess.run([str(VENV_PY), "-c", code], env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout, flush=True)
    assert r.returncode == 0, "فشل الرفع — تحقق من الرمز ومن الاتصال ثم أعد الخلية"
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

UV = globals().get("UV") or shutil.which("uv")

def dir_gb(p: Path) -> float:
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9

targets = [DATA_DIR, ROOT / "raw", ROOT / "venv", SEEDVC / "checkpoints" / "hf_cache"]
labels = ["بيانات التدريب الجاهزة (الأرشيف)", "مصادر خام إضافية", "البيئة المعزولة (تُعاد في 3-8 دقائق)", "مخبأ تنزيل HF"]
print(f"المساحة الحرة الآن: {shutil.disk_usage(ROOT).free/1e9:.1f} GB")
for t, lb in zip(targets, labels):
    s = dir_gb(t)
    print(f"  {t.name:<14} {s:6.2f} GB   ({lb})")

if CFG["CLEAN_AFTER"]:
    for t in targets[:3]:   # البيانات والخام والبيئة — أما نقاط HF فتحذف فقط لو الطلب صريح
        if t.exists():
            shutil.rmtree(t)
            print(f"  حُذف: {t}")
    if (ROOT / ".ready_archive_ok").exists():
        (ROOT / ".ready_archive_ok").unlink()
    if UV:
        sh([UV, "cache", "clean"], check=False)
    else:
        sh([str(VENV_PY), "-m", "pip", "cache", "purge"], check=False)
    print(f"المساحة الحرة بعد التنظيف: {shutil.disk_usage(ROOT).free/1e9:.1f} GB")
    print("كل ما يهم محفوظ على Drive (النقاط + الحزمة + الأرشيف) — لا خسارة دائمة.")
else:
    print('\\nللتفعيل: اجعل  CFG["CLEAN_AFTER"] = True  في الخلية 2 ثم أعد تشغيل هذه الخلية.')'''))

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

(أ) استدلال فوري في هذه الجلسة (البيئة جاهزة الآن):
    في خلية جديدة (نفس أسلوب خلية التحقق 13):
      import subprocess, os
      code = ("import sys, types; sys.path.insert(0, r\\"{SEEDVC}\\");"
              "import os; os.chdir(r\\"{SEEDVC}\\"); import inference as inf;"
              "a = types.SimpleNamespace(source=r\\"<صوت-المتكلم-الأصلي.wav>\\","
              " target=r\\"<صوت-مرجعي-عراقي.wav>\\", output=r\\"./out\\","
              " diffusion_steps=30, length_adjust=1.0, inference_cfg_rate=0.7,"
              " f0_condition=False, auto_f0_adjust=False, semi_tone_shift=0,"
              " checkpoint=r\\"{W}\\", config=r\\"{C}\\", fp16=True);"
              "inf.main(a)")
      subprocess.run([r"{VENV_PY}", "-c", code])

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
| `Runtime disconnected` قديماً | النسخة الأولى كانت تشغّل التدريب كعملية خلفية + مراقب دائم + حلقة انتظار — أنماط «استخدام غير مسموح» في الطبقة المجانية | **أُزيلت كلها في هذه النسخة**: التدريب في مقدمة الخلية 12 — لا عمليات خفية إطلاقاً |
| انقطعت الجلسة أثناء التدريب | حد طبيعي لجلسات Colab | `Run all` — البيئة تُبنى، الأرشيف يُفك من Drive، والخلية 11 تستعيد آخر نقطة، والتدريب يكمل تلقائياً |
| انقطع تنزيل الأرشيف في المنتصف | شبكة | أعد الخلية 7 — `curl -C -` يستكمل من حيث توقف (تحقق الحجم الدقيق بالبايت قبل أي فك) |
| نفدت مساحة القرص | ت accumulation البيانات | خلية التنظيف (17) — أو قلّل مساحة بنسخ أرشيف Drive فقط عند الحاجة |
| `CUDA out of memory` أثناء التدريب | دفعة أكبر من الذاكرة | الخلية 2: `CFG["BATCH"] = 1` ثم أعد الخليتين 3 و12 |
| فشل تنزيل نقطة HF | شبكة/قيود IP | أعد الخلية 10 — التنزيل قابل للاستئناف؛ أو اضبط `HF_ENDPOINT` مرآة في الخلية 2 |
| التدريب بطيء جداً | T4 + مقاطع طويلة | طبيعي على الطبقة المجانية؛ أو قلّل `STEPS`؛ L4 يضاعف السرعة |
| فشل `verify` | مقطع مرجعي شاذ | غير قاتل — الأوزان تُصدَّر على أي حال (الخلية 14) |

**بعد أي انقطاع**: افتح الدفتر ← `Run all` ← كل شيء يستأنف: البيئة (3-8 دقائق)،
البيانات (من Drive: دقائق / من الإنترنت: حتى 20 دقيقة)، النقاط (ثوانٍ)، والتدريب
يكمل من آخر خطوة محفوظة. لا يوجد سيناريو يبدأ فيه الدفتر من الصفر إلا مرة واحدة.

</div>'''))

# ============================================================ MD 9 — final summary
CELLS_C.append(("markdown", r'''<div dir="rtl">

## الخلاصة — ماذا حصلت عليه؟

1. **نموذج مضبوط** `seed_uvit_tat_xlsr_tiny_iraqi.pth` — Seed-VC tiny (25M باراميتر)
   تكيّف على الساعات العراقية الجاهزة، عبر تدريب في مقدمة الخلية بلا أي عمليات خلفية.
2. **ثلاث نسخ دائمة**: Drive (نقاط + حزمة + أرشيف)، tar.gz واحد قابل للتنزيل،
   وHuggingFace Hub (اختياري لكنه الأضمن).
3. **قابلية استئناف كاملة**: كل خلية تتخطى ما أنجز، والأرشيف الجاهز جعل مرحلة
   البيانات دقائق بدل ساعات، والتدريب يستأنف من آخر نقطة.
4. **مسارات نشر جاهزة**: استدلال فوري (استيراد inference)، خدمة رافِدَين الكاملة
   (REST/WebSocket/ويب) على نفس الجلسة، أو أي جهاز GPU آخر.

**الخطوة التالية المقترحة**: شغّل التحقق (13) واستمع للناتج — ثم إن أعجبك، ارفع
النقطة إلى HuggingFace (16) وجرّب خدمة البث الحي (18-ب) بصوتك أنت.

</div>'''))
