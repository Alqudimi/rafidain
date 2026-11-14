# -*- coding: utf-8 -*-
"""Part B: data-stage markdown + code cells 8-12, training markdown + code cells 13-16."""

CELLS_B = []

# ============================================================ MD 4 — data sources
CELLS_B.append(("markdown", r'''<div dir="rtl">

## مرحلة البيانات — المصادر

| المصدر | ما هو | الحجم | ملاحظات |
|---|---|---|---|
| `youssef` | 3,867 مقطعاً عراقياً 22.05kHz مع نصوص | ≈ 720MB | مقاطع قصيرة نظيفة |
| `shiekh` | 827 مقطعاً 16kHz — متحدث واحد | ≈ 640MB | يساعد ثبات المتحدث الواحد |
| `safaa` | 676 ملف wav — متحدثة أنثى جاهزة لـTTS | مئات MB | نظيفة جداً |
| `lahgtna` | الجزء العراقي من Lahgtna v2: ~10,746 مقطعاً / ≈50 ساعة 16kHz مع نصوص لهجية | ≈5.8GB | جديد — يجلب عبر datasets-server بترقيم صفحات (استئناف كامل)؛ قد يتداخل جزئياً مع مصادر YODAS |
| `kharrufa` | 450 مقطعاً استوديو للهجة العراقية 44.1kHz بنصوص مشكولة كاملة | ≈2.1GB تنزيلاً | جديد — يُستخرج منه 683MB ثم يُحذف الأرشيف (Iraqi Dialect TTS Corpus، CC BY 4.0) |
| YODAS‑IQ | مقاطع عراقية مصنّفة من كتالوج YODAS3 (49,194 مقطعاً / 339 ساعة / 2,360 تسجيلاً) | حسب ساعاتك | تُجلب **بتنزيل مُجزَّأ**: نطاق البايتات المطلوب فقط من ملفات tar الضخمة — لهذا 200 ساعة = 12.7GB فقط بدل 22GB+ |
| `custom` | **ملفاتك أنت** — عبر الخلية 10 | حسبك | تسجيلات بموافقة أصحابها، نظيفة قدر الإمكان |

المرحلة 11 (التقطيع والترميز) تحوّل كل ما سبق إلى مقاطع **Opus 24kHz أحادية بطول 1–29.5 ثانية** —
الشكل الذي يقرأه مدرب Seed-VC مباشرة (يفحص المجلد بشكل تكراري: `yodas/`, `youssef/`, `lahgtna/`, `kharrufa/`, `custom/` ...).

فشل أحد المصادر الصغيرة ليس قاتلاً — الخط يتابع بالبقية ويمكن إعادة المحاولة لاحقاً (كل المصادر
تُعلَّم بعلامات `.done` عند الإنجاز فلا يُعاد تنزيلها). على Colab: إن انقطعت الجلسة أثناء التنزيل
عُيد الجلب من حيث توقف عند إعادة تشغيل الخلية — والوقت الميت الوحيد هو إعادة بناء البيئة (الخلية 5).

</div>'''))

# ============================================================ CODE 8 — small sets
CELLS_B.append(("code", r'''# ============================================================ #
#  الخلية ٨ — تنزيل المجموعات العراقية النظيفة الصغيرة         #
#  (youssef / shiekh / safaa من HuggingFace — ≈2-3GB)          #
#  قابلة للاستئناف: أعد التشغيل لتكملة ما فشل فقط              #
# ============================================================ #
import sys
if CFG["SKIP_SMALL"]:
    print("تخطي المجموعات النظيفة الخمس (SKIP_SMALL=True) — إن أردت فعّلها من الخلية 2.")
else:
    import shutil
    free0 = shutil.disk_usage(ROOT).free / 1e9
    print(f"المساحة الحرة قبل التنزيل: {free0:.1f}GB "
          "(المرحلة كلها ≈7.5GB، وأكبر عنصر فيها lahgtna)")
    sh([sys.executable, PIPE, "small",
        "--root", ROOT,
        "--workers", int(CFG["FETCH_WORKERS"])])
    AUD = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".webm"}
    for name, note in (("youssef", "مقاطع قصيرة نظيفة"),
                       ("shiekh", "متحدث ديني واحد"),
                       ("safaa", "صوت أنثى عراقي"),
                       ("lahgtna", "≈50 ساعة عراقية إضافية — جديد"),
                       ("kharrufa", "450 مقطع استوديو — جديد")):
        d = ROOT / "raw" / name
        n = len([p for p in d.iterdir() if p.suffix.lower() in AUD]) if d.exists() else 0
        print(f"  {name:<8}: {n:,} ملفاً | {note}")
    free1 = shutil.disk_usage(ROOT).free / 1e9
    print(f"استُهلك {free0-free1:.1f}GB | الحر الآن {free1:.1f}GB")
    st = ROOT / "raw" / "_lahgtna_state.json"
    if st.exists():
        import json
        s = json.loads(st.read_text())
        fp, fa = len(s.get("failed_pages", [])), len(s.get("failed_audio", []))
        if fp or fa:
            print(f"[ملاحظة] lahgtna: {fp} صفحة و{fa} مقطعاً فشلا مؤقتاً — "
                  "أعد هذه الخلية لإعادة المحاولة فقط (الموجود لا يُعاد)")
print("\n[جاهز] تابع إلى الخلية 9 (YODAS) — أو الخلية 10 لدمج ملفاتك الخاصة.")'''))

# ============================================================ CODE 9 — YODAS
CELLS_B.append(("code", r'''# ============================================================ #
#  الخلية ٩ — جلب بيانات YODAS العراقية (الكتلة الكبرى)        #
#  تنزيل مُجزَّأ متوازٍ من أرشيفات espnet/yodas3                #
#  يستغرق عادة 15-60 دقيقة (حسب الشبكة وساعاتك)                #
#  قابلاً للاستئناف 100%: أعد التشغيل ليكمل من حيث توقف         #
# ============================================================ #
import sys
import shutil

t = str(CFG["YODAS_HOURS"]).strip().lower()
if t == "0":
    print("YODAS معطّل (YODAS_HOURS=0) — تخطّي.")
else:
    free0 = shutil.disk_usage(ROOT).free / 1e9
    print(f"المساحة الحرة قبل الجلب: {free0:.1f}GB")
    sh([sys.executable, PIPE, "yodas",
        "--root", ROOT,
        "--hours", str(CFG["YODAS_HOURS"]),
        "--workers", int(CFG["FETCH_WORKERS"])])
    n = len(list((ROOT / "raw" / "yodas_webm").glob("*.webm"))) if (ROOT / "raw" / "yodas_webm").exists() else 0
    free1 = shutil.disk_usage(ROOT).free / 1e9
    print(f"\nتسجيلات مصدر جاهزة: {n:,} ملف webm | استُهلك {free0-free1:.1f}GB | الحر الآن {free1:.1f}GB")
    state = ROOT / "raw" / "yodas_fetch_state.json"
    if state.exists():
        import json
        failed = json.loads(state.read_text()).get("failed", [])
        if failed:
            print(f"مصادر فشل جلبها: {len(failed)} — أعد هذه الخلية لإعادة محاولة المتبقي، أو تجاهل (بقية البيانات تكفي)")
print("\n[جاهز] تابع إلى الخلية 10 (اختيارية) ثم 11 (التقطيع والترميز).")'''))

# ============================================================ CODE 10 — custom data
CELLS_B.append(("code", r"""# ============================================================ #
#  الخلية ١٠ — (اختيارية) دمج تسجيلاتك الخاصة في التدريب        #
#  على Colab: الشريط الجانبي الأيسر <- أيقونة المجلد <- تنقّل   #
#  إلى /content/iraqi_seedvc/raw/custom <- زر الرفع (أعلى       #
#  اللوحة) ثم أعد تشغيل هذه الخلية                             #
#  ستُقطَّع تلقائياً إلى مقاطع 1-29.5 ثانية وتُرمَّز Opus 24kHz  #
# ============================================================ #
import sys, os, json, time
from multiprocessing import Pool
from pathlib import Path

CUSTOM = ROOT / "raw" / "custom"
CUSTOM.mkdir(parents=True, exist_ok=True)
EXTS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".webm"}
files = [p for p in sorted(CUSTOM.iterdir()) if p.is_file() and p.suffix.lower() in EXTS]
print(f"مجلد بياناتك: {CUSTOM}")
print(f"ملفات صوتية موجودة: {len(files)}")

if not files:
    print('''
لا توجد ملفات بعد — لتفعيل ذلك:
  1) من لوحة الملفات الجانبية في Colab: تنقّل إلى المجلد أعلاه ثم ارفع ملفاتك بزر الرفع
     (يدعم: wav / flac / mp3 / m4a / opus / ogg — الرفع الكبير يستغرق دقائق)
  2) الأفضل: كلام نظيف بلا موسيقى أو ضجيج، 1-30 ثانية أو أطول (تُقطَّع تلقائياً)
  3) أعد تشغيل هذه الخلية، ثم شغّل خلية التقطيع (11) لتشمل بياناتك
(تخطَّ هذه الخلية إن لم يكن لديك بيانات خاصة — مجموعة YODAS + الصغيرة كافية تماماً)''')
else:
    sys.path.insert(0, str(ROOT))
    import iraqi_pipeline as P   # خط البيانات المستخرج في الخلية 7 — نفس منطق التقطيع المجرَّب
    import types
    a = types.SimpleNamespace(out=str(DATA_DIR))
    jobs = []
    for f in files:
        jobs.extend(P._prep_job_src(a, str(f), "custom"))
    print(f"سيُنتج عنها {len(jobs)} مقطعاً (يُتخطى الموجود منها)")
    t0, ok_n, bad_n = time.time(), 0, 0
    with Pool(processes=int(CFG["PREP_WORKERS"] or 0) or min(16, os.cpu_count() or 4)) as pool:
        for out, dur, okk in pool.imap_unordered(P._run_prep_job, jobs, chunksize=16):
            if okk:
                ok_n += 1
            else:
                bad_n += 1
    print(f"تم: {ok_n} مقطعاً ناجحاً | {bad_n} فاشلاً في {time.time()-t0:.0f} ثانية")
    # تحديث فهرس بيانات التدريب (كما يفعل أمر prep نفسه)
    idx = DATA_DIR / "_prep_index.jsonl"
    prev = {}
    if idx.exists():
        for line in idx.read_text().splitlines():
            try:
                r = json.loads(line)
                prev[r["file"]] = r["dur"]
            except Exception:
                pass
    for j in jobs:
        prev[os.path.relpath(j[3], DATA_DIR)] = j[2]
    prev = {f: d for f, d in prev.items() if (DATA_DIR / f).exists()}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    idx.write_text("".join(json.dumps({"file": f, "dur": d}) + "\n" for f, d in sorted(prev.items())))
    print("فهرس بيانات التدريب محدّث — أكمل إلى الخلية 11.")"""))

# ============================================================ CODE 11 — prep/encode
CELLS_B.append(("code", r'''# ============================================================ #
#  الخلية ١١ — التقطيع والترميز: بناء بيانات التدريب النهائية  #
#  كل المقاطع -> Opus 24kHz أحادي، 1-29.5 ثانية                #
#  (يشمل YODAS + المجموعات الخمس + بياناتك إن وضعتها)          #
#  قابلة للاستئناف: الموجود يُتخطى                              #
# ============================================================ #
import sys
import os

workers = int(CFG["PREP_WORKERS"] or 0) or min(16, os.cpu_count() or 4)
sh([sys.executable, PIPE, "prep",
    "--root", ROOT,
    "--out", DATA_DIR,
    "--workers", workers])

n = sum(1 for _ in DATA_DIR.rglob("*.ogg"))
print(f"\nمقاطع التدريب الجاهزة: {n:,} ملف Opus في {DATA_DIR}")
assert n >= 50, f"مقاطع قليلة جداً ({n}) — تأكد أن الخليتين 8 و9 أنزلتا بيانات ثم أعد المحاولة"
print("[جاهز] البيانات مبنية — تابع إلى الخلية 12 (تنزيل نقاط النموذج).")'''))

# ============================================================ CODE 12 — prefetch ckpts
CELLS_B.append(("code", r'''# ============================================================ #
#  الخلية ١٢ — تنزيل نقاط النموذج الأساسية مسبقاً (≈3.5GB)     #
#  DiT base f0 44k + Whisper-base + BigVGAN-44k + RMVPE +      #
#  CAMPPlus + OpenVoice + se_db — كلها في checkpoints/seed-vc  #
#  (يمنع تعطل بدء التدريب لاحقاً بسبب التنزيل، وقابل للاستئناف)#
# ============================================================ #
import sys
sh([sys.executable, PIPE, "prefetch", "--repo", SEEDVC])
import shutil
ck = SEEDVC / "checkpoints"
total = sum(f.stat().st_size for f in ck.rglob("*") if f.is_file()) / 1e9 if ck.exists() else 0
print(f"\nحجم مخزون النقاط: {total:.2f}GB في {ck}")
print("[جاهز] كل شيء مطلوب للتدريب موجود — تابع إلى مرحلة التدريب.")'''))

# ============================================================ MD 5 — training explanation
CELLS_B.append(("markdown", r'''<div dir="rtl">

## مرحلة التدريب — كيف تعمل؟ (بمنطق Colab)

- **نقطة البداية دائماً** الأوزان المنشورة `DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth`
  من `Plachta/Seed-VC` — **نموذج base مع شرط النبرة F0 وترميز 44.1kHz** (نحو 78M باراميتر): يحافظ على
  النبرة العراقية بدقة أعلى بكثير من نموذج tiny. لا تدريب من الصفر أبداً؛ وإن وُجدت نقطة سابقة في مجلد
  التشغيل يُستأنف منها تلقائياً (يُحتفظ بآخر نقطتين `DiT_epoch_*_step_*.pth` — سلوك المدرب الرسمي).
- **السجل**: سطر `epoch E, step S, loss: L` كل 10 خطوات؛ نقطة حفظ كل `SAVE_EVERY` خطوة (500 افتراضياً)؛
  وفي النهاية يُكتب `ft_model.pth` (أوزان DiT المضبوطة — وهي كل ما يلزم للنشر والاستدلال).
- **الإعدادات الموصى بها من دليل المستودع (FINETUNING.md)**: دفعة 1 لذاكرة 16GB (T4) — النموذج أكبر
  والدفعة الصغيرة أمتن مع صدمات الدفعة؛ **3000–6000 خطوة لتكيّف عام مع اللهجة** (3000 افتراضياً هنا)؛
  معدل التعلم مثبّت في الإعداد الرسمي (1e-5 مع ExponentialLR) — لا ترفعه للضبط الدقيق.
- **لماذا التدريب داخل النواة؟** الخلية 14 تستورد وحدة `train.py` الرسمية وتشغلها في نواة الدفتر نفسها
  (بلا أي عملية ابن) — فالتقدم الحي (شريط tqdm والخسارة) يظهر في الخلية مباشرة، والسجل نفسه يُكتب في
  `train_console.log` للمراقبة بالخليتين 15/16. لو قُطعت الجلسة، تُستأنف النقاط من Drive بالخلية 13.
- **مرآة Drive الحية**: أثناء التدريب يعمل خيط خفيف يزامن كل نقطة جديدة إلى
  `MyDrive/rafidain_seedvc/runs/<اسم-التشغيل>/` خلال دقيقة من حفظها — أي أن أقصى خسارة عند أي انقطاع
  جلسة = آخر `SAVE_EVERY` خطوة فقط (500 خطوة ≈ دقائق معدودة على T4).
- **انقطعت الجلسة؟** افتح الدفتر ← **Run all**: تُثبَّت الاعتماديات داخل النواة (دقائق عبر uv)، وتستعيد
  الخلية 13 النقاط من Drive، ويستأنف التدريب من آخر نقطة حفظ. البيانات الخام إن فُقد القرص المحلي
  تُجلب من جديد (استئناف كامل).

مدة تقديرية: 3000 خطوة على **T4 في حدود 4–7 ساعات تقريباً** (النموذج الأكبر أدق لكنه أبطأ)، وعلى
**L4 نحو نصف ذلك**. راقب معدل الخطوات في خلية المراقبة بعد أول دقيقة لتقدير زمنك الفعلي بدقة.

</div>'''))

# ============================================================ CODE 13 — restore from Drive
CELLS_B.append(("code", r'''# ============================================================ #
#  الخلية ١٣ — استعادة نقاط التدريب من Drive (بعد أي انقطاع)   #
#  تعيد أحدث نقاط التفتيش + السجلات من نسخة Drive الاحتياطية   #
#  إلى القرص المحلي فيستأنف التدريب من حيث توقف                #
#  (خلية استباقية: لا تفعل شيئاً إن كان المحلي أحدث أو لا نسخة)#
# ============================================================ #
import shutil, re
from pathlib import Path

DRIVE_OK   = globals().get("DRIVE_OK", False)
DRIVE_ROOT = globals().get("DRIVE_ROOT", None)

def _step_of(p: Path):
    m = re.search(r"step_(\d+)", p.stem)
    return int(m.group(1)) if m else -1

loc_ckpts = sorted(RUN_DIR.glob("DiT_epoch_*_step_*.pth"), key=_step_of)
loc_ft    = RUN_DIR / "ft_model.pth"
drv       = (DRIVE_ROOT / "runs" / CFG["RUN_NAME"]) if (DRIVE_OK and DRIVE_ROOT) else None

if loc_ckpts or loc_ft.exists():
    last = f" (حتى الخطوة {_step_of(loc_ckpts[-1])})" if loc_ckpts else ""
    print(f"[محلي] توجد نقاط محلية أصلاً{last} — لا حاجة للاستعادة.")
    print("        (لفرض الاستعادة من Drive احذف مجلد التشغيل المحلي ثم أعد هذه الخلية)")
elif drv and drv.exists():
    files = [p for p in drv.iterdir() if p.is_file()] if drv.is_dir() else []
    if not files:
        print(f"[Drive] لا توجد نسخة احتياطية سابقة في {drv} — تشغيل جديد كلياً.")
    else:
        print(f"[Drive] استعادة {len(files)} ملفاً من {drv} ...")
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        for f in files:
            print(f"  + {f.name}  ({f.stat().st_size/1e6:.1f} MB)")
            shutil.copy2(f, RUN_DIR / f.name)
        ck = sorted(RUN_DIR.glob("DiT_epoch_*_step_*.pth"), key=_step_of)
        if ck:
            print(f"\n[تم] أُعيدت أحدث نقطة إلى الخطوة {_step_of(ck[-1])} — التدريب سيكمل منها (الخلية 14).")
        elif (RUN_DIR / "ft_model.pth").exists():
            print("\n[تم] النموذج النهائي ft_model.pth مستعاد — يمكنك القفز إلى التحقق (الخلية 17).")
        else:
            print("\n[تم] استُعيدت السجلات فقط — سيبدأ التدريب من الصفر.")
else:
    print("[استعادة] لا نسخة Drive (غير موصول أو لا سجل سابق) — تشغيل جديد كلياً.")'''))

# ============================================================ CODE 14 — in-kernel training + drive mirror
CELLS_B.append(("code", r"""# ============================================================ #
#  الخلية ١٤ — إطلاق التدريب داخل نواة الدفتر مباشرة           #
#  (استيراد وحدة train.py الرسمية — بلا أي عملية ابن مستقلة)   #
#  التقدم يظهر حياً في الخلية + يُسجل في train_console.log     #
#  ومرآة Drive تنسخ كل نقطة فور حفظها أثناء التدريب           #
# ============================================================ #
import subprocess, os, time, sys, threading
from pathlib import Path

LOG  = RUN_DIR / "train_console.log"
FT   = RUN_DIR / "ft_model.pth"
CFG_MODEL = SEEDVC / "configs" / "presets" / "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"

# ---- 0) كتابة مزامن النسخ إلى Drive (stdlib فقط — يعمل بأي بايثون) ----
SYNC_SCRIPT = ROOT / "drive_sync_loop.py"
SYNC_SCRIPT.write_text(r'''
import os, sys, time, shutil
from pathlib import Path

RUN_DIR  = Path(sys.argv[1])   # مجلد التشغيل المحلي
DRV_DIR  = Path(sys.argv[2])   # مجلد النسخ على Drive
TRAIN_PID = int(sys.argv[3])   # PID عملية التدريب (0 = بلا فحص حياة)
INTERVAL = int(os.environ.get("SYNC_INTERVAL", "300"))
KEEP     = int(os.environ.get("SYNC_KEEP", "2"))
ONCE     = "--once" in sys.argv

def _step(p):
    try:    return int(p.stem.rsplit("_", 1)[-1])
    except Exception: return -1

def _alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0); return True
    except Exception:
        return False

def sync_once():
    if not RUN_DIR.exists():
        return 0
    DRV_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    # نقاط DiT: الأحدث KEEP فقط (كل نقطة تحمل net+optimizer+scheduler)
    ck = sorted(RUN_DIR.glob("DiT_epoch_*_step_*.pth"), key=_step)
    for p in ck[-KEEP:]:
        d = DRV_DIR / p.name
        if (not d.exists()) or (d.stat().st_mtime < p.stat().st_mtime):
            shutil.copy2(p, d); n += 1
    for p in sorted(DRV_DIR.glob("DiT_epoch_*_step_*.pth"), key=_step)[:-KEEP]:
        try: p.unlink()
        except Exception: pass
    # الملفات الصغيرة: دائماً
    for name in ("ft_model.pth", "train_console.log", "loss_curve.png",
                 "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"):
        s = RUN_DIR / name
        if s.exists():
            d = DRV_DIR / name
            if (not d.exists()) or (d.stat().st_mtime < s.stat().st_mtime):
                shutil.copy2(s, d); n += 1
    return n

if ONCE:
    print(f"[sync] {sync_once()} file(s) updated on Drive")
    sys.exit(0)

print(f"[sync] drive backup watcher: every {INTERVAL}s, keep last {KEEP} ckpts -> {DRV_DIR}", flush=True)
while True:
    try:
        n = sync_once()
        if n:
            print(f"[sync] {time.strftime('%H:%M:%S')} {n} file(s) -> Drive", flush=True)
    except Exception as e:
        print(f"[sync] skip cycle ({e})", flush=True)
    if not _alive(TRAIN_PID):
        try: sync_once()
        except Exception: pass
        print("[sync] training process ended - final sync done, watcher exits", flush=True)
        break
    time.sleep(INTERVAL)
''', encoding="utf-8")

BATCH = int(CFG.get("_BATCH") or 1)
MAXS  = int(CFG["STEPS"])

# ---- 1) إطلاق التدريب داخل النواة مباشرة (import train — بلا أي عملية ابن) ----
# لماذا داخل النواة؟ التدريب كعملية ابن مستقلة يفقد سياق النواة عند إعادة تشغيلها
# ويعقّد الاستئناف؛ أما بالاستيراد المباشر فالتقدم يظهر حياً في الخلية نفسها،
# والنقاط تُحفظ على القرص فتُستعاد بأمان من أي جلسة جديدة (الخلية 13).
if FT.exists():
    print(f"النموذج النهائي موجود أصلاً: {FT}")
    print("لتدريب جديد كلياً: غيّر RUN_NAME في الخلية 2 (أو احذف ft_model.pth ثم أعد هذه الخلية).")
else:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # ---- مرآة Drive الحية: خيط خفيف يزامن كل نقطة أثناء التدريب داخل النواة ----
    _DRV = (DRIVE_ROOT / "runs" / CFG["RUN_NAME"]) if (globals().get("DRIVE_OK") and globals().get("DRIVE_ROOT") and CFG.get("DRIVE_BACKUP", True)) else None
    if _DRV:
        _keep = str(int(CFG.get("DRIVE_CKPT_KEEP", 2)))
        def _drive_mirror(stop_evt, drv=_DRV, keep=_keep):
            # يستدعي drive_sync_loop.py --once كل 60 ثانية حتى نهاية التدريب
            import subprocess as _sp
            env_s = dict(ENV); env_s["SYNC_KEEP"] = keep
            while not stop_evt.wait(60):
                try:
                    _sp.run([sys.executable, str(SYNC_SCRIPT), str(RUN_DIR), str(drv), "0", "--once"],
                            capture_output=True, text=True, timeout=300, env=env_s)
                except Exception:
                    pass
        _stop_evt = threading.Event()
        _mirror = threading.Thread(target=_drive_mirror, args=(_stop_evt,), daemon=True)
        _mirror.start()
        print(f"[train] مرآة Drive مفعّلة: كل نقطة تُنسخ خلال دقيقة إلى {_DRV} (أبقِ آخر {int(CFG.get('DRIVE_CKPT_KEEP', 2))})")
    else:
        _stop_evt = None
        print("[train] Drive غير موصول — النقاط تبقى محلية فقط.")
    print(f"[train] بدء الضبط الدقيق: {MAXS} خطوة، دفعة {BATCH}، حفظ كل {int(CFG['SAVE_EVERY'])}")
    print(f"[train] التشغيل داخل نواة الدفتر مباشرة (import train.Trainer) — بلا أي عملية ابن")
    # train.py يعمل من مجلد المستودع (مساراته نسبية: configs/, checkpoints/, runs/)
    import os as _os, sys as _sys
    _os.chdir(SEEDVC)
    _sys.path.insert(0, str(SEEDVC))
    import train as SEED_TRAIN   # وحدة المدرب الرسمية — نفسها التي تستخدمها train.py

    trainer = SEED_TRAIN.Trainer(
        config_path=str(CFG_MODEL),
        pretrained_ckpt_path=None,     # None = يكتشف أحدث نقطة في مجلد التشغيل تلقائياً (استئناف)
        data_dir=str(DATA_DIR),
        run_name=CFG["RUN_NAME"],
        batch_size=BATCH,
        steps=MAXS,
        max_epochs=1000,               # السقف الفعلي هو MAXS (يتوقف عنده)
        save_interval=int(CFG["SAVE_EVERY"]),
        num_workers=int(CFG["TRAIN_WORKERS"]),
        device="cuda:0",
    )
    # توجيه مخرجات السجلات إلى ملف السجل أيضاً (نفس ملف المراقبة في الخلية 15)
    # ملاحظة: نحفظ مرجع sys.stdout الأصلي أولاً — print تكتب فيه مباشرة فيمنع أي تكرار
    import functools
    _logfh = open(LOG, "a", buffering=1)
    _old_out, _old_err = _sys.stdout, _sys.stderr
    _orig_print = functools.partial(print, file=_old_out)
    class _Tee:
        def write(self, s):
            _logfh.write(s)
            _old_out.write(s)
        def flush(self):
            _logfh.flush(); _old_out.flush()
        def isatty(self):
            return False
    _sys.stdout = _Tee(); _sys.stderr = _Tee()
    try:
        trainer.train()
    finally:
        _sys.stdout, _sys.stderr = _old_out, _old_err
        _logfh.close()
        if _stop_evt is not None:
            _stop_evt.set()   # أوقف مرآة Drive
    print(f"\n[train] اكتمل — النموذج النهائي: {RUN_DIR / 'ft_model.pth'}")

# ---- 2) مزامنة نهائية فورية إلى Drive (التدريب انتهى داخل النواة — لا حاجة لمراقب خلفي) ----
DRV_RUN = (DRIVE_ROOT / "runs" / CFG["RUN_NAME"]) if (globals().get("DRIVE_OK") and globals().get("DRIVE_ROOT")) else None
if DRV_RUN and SYNC_SCRIPT.exists():
    sh([sys.executable, str(SYNC_SCRIPT), str(RUN_DIR), str(DRV_RUN), "0", "--once"], check=False)
    print(f"[نسخ احتياطي] نُسخت النقاط والسجلات إلى Drive: {DRV_RUN}")
elif not DRV_RUN:
    print("[نسخ احتياطي] Drive غير موصول — النقاط تبقى محلية فقط.")

if FT.exists():
    print("\nالخطوة التالية: خلية «التحقق» (17) للاستماع إلى النموذج، ثم التصدير (18).")"""))

# ============================================================ CODE 15 — monitor
CELLS_B.append(("code", r'''# ============================================================ #
#  الخلية ١٥ — مراقبة التدريب (أعد تشغيلها متى شئت)            #
#  تقدم + معدل + زمن متبقٍ + منحنى الخسارة + آخر السجل          #
#  + حالة النسخ الاحتياطي إلى Drive                            #
#  (تسميات الرسم بالإنجليزية لأن matplotlib لا يشكّل            #
#   الحروف العربية المتصلة بشكل صحيح)                          #
# ============================================================ #
import re, os, time, subprocess, sys
from pathlib import Path
from IPython.display import display, Image as IPyImage

LOG  = RUN_DIR / "train_console.log"
FT   = RUN_DIR / "ft_model.pth"
STEP_RE = re.compile(r"epoch (\d+), step (\d+), loss: ([0-9.eE+-]+)")
MAXS = int(CFG["STEPS"])

def _running_in_kernel():
    # التدريب داخل النواة: كائن المدرب حي في ذاكرة الجلسة
    t = globals().get("trainer")
    return t is not None and getattr(t, "iters", None) is not None

lines = LOG.read_text(errors="replace").splitlines() if LOG.exists() else []
pts = []
for ln in lines:
    m = STEP_RE.search(ln)
    if m:
        pts.append((int(m.group(2)), float(m.group(3))))

if FT.exists():
    status = "منجز — ft_model.pth موجود (انتقل إلى خلية التحقق 17)"
elif _running_in_kernel():
    status = "يعمل الآن (داخل النواة)"
elif pts:
    status = "متوقف قبل الاكتمال (راجع آخر السجل — أعد الخلية 14 للاستئناف من آخر نقطة)"
else:
    status = "لم يبدأ بعد"

steps = pts[-1][0] if pts else 0
start = LOG.stat().st_mtime if LOG.exists() else time.time()
elapsed = max(time.time() - start, 1.0)
rate = steps / elapsed if steps else 0.0
eta_min = (MAXS - steps) / rate / 60 if rate > 0 else float("nan")
pct = min(100.0 * steps / MAXS, 100.0) if MAXS else 0.0
filled = int(30 * pct / 100)
bar = "#" * filled + "-" * (30 - filled)

print(f"الحالة: {status}")
print(f"[{bar}] {pct:5.1f}%   step {steps}/{MAXS}   معدل {rate:.2f} خطوة/ث   زمن منقضٍ {elapsed/60:.0f} د   متبقٍ ≈ {eta_min:.0f} د")
if pts:
    print(f"آخر خسارة (EMA): {pts[-1][1]:.4f} عند الخطوة {steps}")

# ---- منحنى الخسارة ----
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
    display(IPyImage(filename=str(chart)))
    print(f"(حُفظ أيضاً بشكل دائم: {chart})")

# ---- آخر السجل ----
print("\n— آخر 12 سطراً من train_console.log —")
for ln in lines[-12:]:
    print("  " + ln)
ckpts = sorted(RUN_DIR.glob("DiT_epoch_*_step_*.pth"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
if ckpts:
    print(f"\nنقاط التفتيش المحفوظة (آخر 3): " + ", ".join(c.name for c in ckpts[-3:]))

# ---- حالة النسخ الاحتياطي إلى Drive ----
SYNC_SCRIPT = ROOT / "drive_sync_loop.py"
DRV_RUN = (globals().get("DRIVE_ROOT") / "runs" / CFG["RUN_NAME"]) if (globals().get("DRIVE_OK") and globals().get("DRIVE_ROOT")) else None
if DRV_RUN and SYNC_SCRIPT.exists():
    try:
        r = subprocess.run([sys.executable, str(SYNC_SCRIPT), str(RUN_DIR), str(DRV_RUN), "0", "--once"],
                           capture_output=True, text=True, timeout=900)
        drv_ck = sorted(DRV_RUN.glob("DiT_epoch_*_step_*.pth"), key=lambda p: int(p.stem.rsplit("_", 1)[-1])) if DRV_RUN.exists() else []
        print(f"\n[Drive] النسخة الاحتياطية: {len(drv_ck)} نقطة على Drive"
              + (f" (أحدثها حتى الخطوة {int(drv_ck[-1].stem.rsplit('_', 1)[-1])})" if drv_ck else " — تُنشئ عند أول حفظ نقطة")
              + f" | {r.stdout.strip()}")
    except Exception as e:
        print(f"\n[Drive] تعذرت المزامنة اللحظية ({e}) — المراقب الخلفي (الخلية 14) يكفي عادة.")'''))

# ============================================================ CODE 16 — auto-wait
CELLS_B.append(("code", r'''# ============================================================ #
#  الخلية ١٦ — (اختيارية) مراقبة تقدم التدريب أثناء عمله        #
#  تُشغَّل في خلية ثانية أثناء دوران التدريب في الخلية 14:      #
#  تحديث حي كل 30 ثانية + مزامنة Drive كل دورة                 #
#  ملاحظة: النشاط هنا لا يبقي الجلسة حية بحد ذاته — الخلية 14  #
#  النشطة (التدريب داخل النواة) هي التي تمنع القطع الخامل      #
# ============================================================ #
import re, os, time, subprocess, sys
from pathlib import Path
from IPython.display import clear_output

LOG  = RUN_DIR / "train_console.log"
FT   = RUN_DIR / "ft_model.pth"
STEP_RE = re.compile(r"epoch (\d+), step (\d+), loss: ([0-9.eE+-]+)")
MAXS = int(CFG["STEPS"])
SYNC_SCRIPT = ROOT / "drive_sync_loop.py"
DRV_RUN = (globals().get("DRIVE_ROOT") / "runs" / CFG["RUN_NAME"]) if (globals().get("DRIVE_OK") and globals().get("DRIVE_ROOT")) else None

def _running_in_kernel():
    t = globals().get("trainer")
    return t is not None and getattr(t, "iters", None) is not None

def _drive_sync():
    if DRV_RUN and SYNC_SCRIPT.exists():
        try:
            subprocess.run([sys.executable, str(SYNC_SCRIPT), str(RUN_DIR), str(DRV_RUN), "0", "--once"],
                           capture_output=True, text=True, timeout=600)
        except Exception:
            pass

deadline = time.time() + 12 * 3600   # مهلة قصوى 12 ساعة (سقف جلسة Colab)
while time.time() < deadline:
    clear_output(wait=True)
    lines = LOG.read_text(errors="replace").splitlines() if LOG.exists() else []
    steps, loss = 0, None
    for ln in lines:
        m = STEP_RE.search(ln)
        if m:
            steps, loss = int(m.group(2)), float(m.group(3))
    pct = min(100.0 * steps / MAXS, 100.0)
    bar = "#" * int(30 * pct / 100)
    msg = f"[{bar:<30}] {pct:5.1f}%   step {steps}/{MAXS}"
    if loss is not None:
        msg += f"   loss {loss:.4f}"
    print(msg, flush=True)
    _drive_sync()
    if FT.exists():
        print("\nاكتمل التدريب — ft_model.pth جاهز. انتقل إلى خلية التحقق (17).")
        _drive_sync()
        break
    if not _running_in_kernel():
        print("\nالتدريب متوقف قبل الاكتمال (الخلية 14 انتهت أو قُطعت).")
        print("شغّل خلية المراقبة (15) لرؤية آخر السجل — أعد الخلية 14 للاستئناف من آخر نقطة.")
        break
    time.sleep(30)
else:
    print("انتهت مهلة المراقبة (12 ساعة) — استخدم الخلية 15 للفحص اليدوي.")
    print("إن كانت الجلسة ستُقطع قريباً: نقاطك محفوظة على Drive وتُستعاد بالخلية 13.")'''))
