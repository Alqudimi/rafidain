# -*- coding: utf-8 -*-
"""Part B: data markdown + code cells 7-10 (ready archive, custom data, stats, prefetch).

v2: the ready-made archive replaces the small-sets + YODAS + prep cells entirely.
The archive is downloaded once (resumable), verified by exact byte size,
extracted straight into DATA_DIR, and never re-downloaded/re-processed:
a stamp file (.ready_archive_ok) records the fingerprint + extraction time,
and a Drive copy is reused if present to skip the 4.9GB download.
prefetch is invoked by importing iraqi_pipeline inside the venv python.
"""

CELLS_B = []

# ============================================================ MD 4 — data
CELLS_B.append(("markdown", r'''<div dir="rtl">

## ٣) البيانات الجاهزة — أرشيف واحد بدل جلب المصادر ومعالجتها

النسخة الأولى من هذا الدفتر كانت تنزّل من خمسة مصادر (YODAS وغيرها) ثم تقطّع
وتُرمّز كل شيء من الصفر — ساعتان أو أكثر قبل أول خطوة تدريب. هنا بدل ذلك كله
**أرشيف واحد مُجهَّز مسبقاً**: `rafidain_current_sources_ready.tar.gz` (4.89GB) يحتوي
مجلد `iraqi_finetune/` وفيه المقاطع النهائية جاهزة (Opus 24kHz أحادي، 1–29.5 ثانية)
من كل المصادر العراقية — أي بالضبط ما ينتجه أمرا `small`+`yodas`+`prep` في خط
البيانات، لكن جاهزاً.

**كيف يتعامل الدفتر مع الأرشيف بلا تكرار؟**
1. إن وُجدت بصمة `.ready_archive_ok` (جلسة سابقة على نفس الجهاز) — تخطٍ كامل فوري.
2. وإلا إن وُجدت نسخة الأرشيف على Drive — تُنسخ منه مباشرة (أسرع وأضمن من الإنترنت).
3. وإلا يُنزَّل من الرابط المباشر بتنزيل **قابل للاستئناف** (curl -C -) — انقطاع
   التنزيل لا يعيد البداية.
4. قبل الفك يتحقق الدفتر من **الحجم الدقيق بالبايت** — نصف ملف مضغوط لا يُفك أبداً.
5. بعد الفك تُحذف نسخة tar.gz المحلية (المساحة تهم)، مع خيار نسخ الأرشيف إلى Drive
   لجلسة انقطاع قادمة أسرع (اختياري — مُفعّل افتراضياً إن توفرت المساحة).

> خلاصة: في أسوأ الحالات (جلسة أولى بلا Drive) تدفع ثمن التنزيل مرة واحدة؛ وفي كل
> الحالات الأخرى البيانات تظهر جاهزة خلال دقيقة إلى دقائق معدودة.

</div>'''))

# ============================================================ CODE 7 — ready archive
CELLS_B.append(("code", r'''# ============================================================ #
#  الخلية ٧ — البيانات الجاهزة: تنزيل الأرشيف وفكّه (مرة واحدة) #
#  rafidain_current_sources_ready.tar.gz (4.89GB)              #
#  بلا إعادة تنزيل من المصادر وبلا أي إعادة معالجة            #
# ============================================================ #
import os, shutil, subprocess, time, json
from pathlib import Path

URL    = CFG["READY_ARCHIVE_URL"]
EXPECT = int(CFG["READY_ARCHIVE_SIZE"])          # الحجم الدقيق بالبايت (تحقق الاكتمال)
INNER  = CFG["READY_DIR_NAME"]                   # المجلد الجذري داخل الأرشيف
DST    = READY_TGZ                               # /content/iraqi_seedvc/rafidain_..._ready.tar.gz

def archive_complete(p: Path) -> bool:
    return p.exists() and p.stat().st_size == EXPECT

def extract_ready():
    """فك الأرشيف: محتوى iraqi_finetune/ يصبح بيانات التدريب في DATA_DIR."""
    t0 = time.time()
    print(f"فك الأرشيف إلى {DATA_DIR} ... (2-4 دقائق عادة)")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # ننزع الضغط في مجلد مؤقت ثم ننقل محتوى iraqi_finetune — أمان من أي اختلاف هيكلي
    tmp = ROOT / "_ready_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    sh(["tar", "-xzf", str(DST), "-C", str(tmp)])
    inner_dir = tmp / INNER
    assert inner_dir.is_dir(), f"بنية الأرشيف غير متوقعة: {INNER}/ غير موجود"
    moved = 0
    for src in inner_dir.iterdir():
        dst = DATA_DIR / src.name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))
        moved += 1
    shutil.rmtree(tmp)
    n = sum(1 for _ in DATA_DIR.rglob("*.ogg"))
    hours = 0.0
    idx = DATA_DIR / "_prep_index.jsonl"
    if idx.exists():
        for line in idx.read_text().splitlines():
            try:
                hours += json.loads(line)["dur"] / 3600
            except Exception:
                pass
    READY_STAMP.write_text(json.dumps({
        "archive": DST.name, "size": EXPECT,
        "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ogg_files": n, "indexed_hours": round(hours, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"فُكّت {moved} مدخلاً | {n:,} ملف Opus | فهرس: {hours:.1f} ساعة | {time.time()-t0:.0f} ثانية")

# ---------- 0) منجز أصلاً؟ ----------
if READY_STAMP.exists() and any(DATA_DIR.rglob("*.ogg")):
    n = sum(1 for _ in DATA_DIR.rglob("*.ogg"))
    print(f"[تخطّي] البيانات الجاهزة موجودة أصلاً: {n:,} ملف Opus في {DATA_DIR}")
    print("        (لبناء البيانات من المصدر بدل الأرشيف: احذف " + str(READY_STAMP) + " واستخدم سكربتات المستودع)")
# ---------- 1) نسخة محلية كاملة من تنزيل سابق؟ ----------
elif archive_complete(DST):
    print(f"[عثرنا] أرشيف محلي كامل من محاولة سابقة ({DST.stat().st_size/1e9:.2f}GB) — فك مباشرة بلا تنزيل")
    extract_ready()
else:
    # ---------- 2) نسخة Drive؟ (توفر إعادة التنزيل بعد انقطاع الجلسة) ----------
    drv_arc = (drive_data_dir() / DST.name) if drive_data_dir() else None
    used_drive = False
    if drv_arc and archive_complete(drv_arc):
        print(f"[Drive] نسخة كاملة من الأرشيف موجودة على Drive — النسخ منها أسرع من التنزيل ...")
        t0 = time.time()
        shutil.copy2(drv_arc, DST)
        print(f"[Drive] نُسخ في {time.time()-t0:.0f} ثانية")
        used_drive = True
    # ---------- 3) التنزيل القابل للاستئناف ----------
    if not archive_complete(DST):
        if DST.exists():
            print(f"[استئناف] يوجد ملف جزئي ({DST.stat().st_size/1e9:.2f}GB من {EXPECT/1e9:.2f}GB) — يُستكمل من حيث توقف")
        free_gb = shutil.disk_usage(ROOT).free / 1e9
        assert free_gb > EXPECT/1e9 + 7, f"المساحة لا تكفي (تحتاج {(EXPECT/1e9+7):.1f}GB)"
        print(f"تنزيل الأرشيف الجاهز ({EXPECT/1e9:.2f}GB) — عادة 5-20 دقيقة حسب الشبكة ...")
        t0 = time.time()
        # curl -C - : استئناف تلقائي؛ المحاولات المتكررة تكسر الاتصالات المعلقة
        rc = sh(["bash", "-c",
                 f"curl -L -C - --retry 8 --retry-delay 3 --connect-timeout 30 "
                 f"-o '{DST}' '{URL}'"], check=False)
        if not archive_complete(DST):
            raise RuntimeError(
                f"التنزيل لم يكتمل (حجم الملف {DST.stat().st_size if DST.exists() else 0:,} من {EXPECT:,} بايت)\n"
                "أعد هذه الخلية — التنزيل قابل للاستئناف وسيتابع من حيث توقف.")
        print(f"[تم] اكتمل التنزيل في {(time.time()-t0)/60:.1f} دقيقة")
    extract_ready()

    # ---------- 4) نسخ الأرشيف إلى Drive (سرعة الجلسة القادمة) — بعد إتمام كل شيء ----------
    if (not used_drive) and drive_data_dir() and CFG.get("DRIVE_BACKUP", True):
        try:
            ddu = shutil.disk_usage(DRIVE_ROOT)
            if ddu.free > EXPECT/1e9 + 1.5:
                print(f"\n[Drive] حفظ نسخة الأرشيف لجلسات قادمة ({EXPECT/1e9:.2f}GB) ...")
                t0 = time.time()
                shutil.copy2(DST, drive_data_dir() / DST.name)
                print(f"[Drive] حُفظت في {time.time()-t0:.0f} ثانية — الجلسة القادمة ستستغني عن التنزيل كلياً")
            else:
                print(f"\n[Drive] لا مساحة كافية لنسخة الأرشيف ({ddu.free/1e9:.1f}GB حر) — تخطّي (لا يؤثر)")
        except Exception as e:
            print(f"[Drive] تعذر حفظ نسخة الأرشيف ({e}) — لا يؤثر على العمل")

# ---------- 5) حذف المضغوط المحلي (وفّر 4.9GB — البصمة تحمي من إعادة التنزيل) ----------
if DST.exists() and READY_STAMP.exists() and not any(ROOT.glob("_keep_archive")):
    DST.unlink()
    print(f"[مساحة] حُذف المضغوط المحلي — الحر الآن {shutil.disk_usage(ROOT).free/1e9:.1f}GB")

print(f"\n[جاهز] بيانات التدريب: {DATA_DIR}")
print("        تابع إلى خلية بياناتك الخاصة (8، اختيارية) أو إحصاءات البيانات (9).")'''))

# ============================================================ CODE 8 — custom data
CELLS_B.append(("code", r'''# ============================================================ #
#  الخلية ٨ — (اختيارية) دمج تسجيلاتك الخاصة في التدريب        #
#  على Colab: الشريط الجانبي الأيسر <- أيقونة المجلد <- تنقّل   #
#  إلى /content/iraqi_seedvc/raw/custom <- زر الرفع (أعلى      #
#  اللوحة) ثم أعد تشغيل هذه الخلية                             #
#  ستُقطَّع تلقائياً إلى مقاطع 1-29.5 ثانية وتُرمَّز Opus 24kHz  #
#  (نفس منطق خط البيانات الرسمي — عبر الاستيراد لا سطر الأوامر)#
# ============================================================ #
import sys, os, json, time, types
from multiprocessing import Pool
from pathlib import Path

CUSTOM = ROOT / "raw" / "custom"
CUSTOM.mkdir(parents=True, exist_ok=True)
EXTS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".webm"}
files = [p for p in sorted(CUSTOM.iterdir()) if p.is_file() and p.suffix.lower() in EXTS]
print(f"مجلد بياناتك: {CUSTOM}")
print(f"ملفات صوتية موجودة: {len(files)}")

if not files:
    print(R"""
لا توجد ملفات بعد — لتفعيل ذلك:
  1) من لوحة الملفات الجانبية في Colab: تنقّل إلى المجلد أعلاه ثم ارفع ملفاتك بزر الرفع
     (يدعم: wav / flac / mp3 / m4a / opus / ogg — الرفع الكبير يستغرق دقائق)
  2) الأفضل: كلام نظيف بلا موسيقى أو ضجيج، 1-30 ثانية أو أطول (تُقطَّع تلقائياً)
  3) أعد تشغيل هذه الخلية — سيلتقطها التدريب تلقائياً في الخلية 12
(تخطَّ هذه الخلية إن لم يكن لديك بيانات خاصة — الأرشيف الجاهز كافٍ تماماً)""")
else:
    # التقطيع عبر وحدة خط البيانات المستوردة (الخلية 6) — بلا سطر أوامر
    a = types.SimpleNamespace(out=str(DATA_DIR))
    jobs = []
    for f in files:
        jobs.extend(PIPELINE._prep_job_src(a, str(f), "custom"))
    print(f"سيُنتج عنها {len(jobs)} مقطعاً (يُتخطى الموجود منها)")
    t0, ok_n, bad_n = time.time(), 0, 0
    with Pool(processes=min(16, os.cpu_count() or 4)) as pool:
        for out, dur, okk in pool.imap_unordered(PIPELINE._run_prep_job, jobs, chunksize=16):
            if okk:
                ok_n += 1
            else:
                bad_n += 1
    print(f"تم: {ok_n} مقطعاً ناجحاً | {bad_n} فاشلاً في {time.time()-t0:.0f} ثانية")
    # تحديث فهرس بيانات التدريب (نفس منطق cmd_prep في الوحدة)
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
    print("فهرس بيانات التدريب محدّث — أكمل إلى الخلية 9.")'''))

# ============================================================ CODE 9 — data stats
CELLS_B.append(("code", r'''# ============================================================ #
#  الخلية ٩ — إحصاءات بيانات التدريب (من فهرس الأرشيف الجاهز)  #
#  تقرير الساعات/الملفات لكل مصدر — بلا فحص صوتيات ثقيل        #
# ============================================================ #
import json
from collections import defaultdict
from pathlib import Path

idx = DATA_DIR / "_prep_index.jsonl"
assert idx.exists(), "لا يوجد فهرس بيانات — شغّل خلية الأرشيف الجاهز (7) أولاً"

per, tot_h, tot_n = defaultdict(lambda: [0, 0.0]), 0.0, 0
for line in idx.read_text().splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    if not (DATA_DIR / r["file"]).exists():
        continue
    src = r["file"].split("/")[0]
    per[src][0] += 1
    per[src][1] += r["dur"] / 3600
    tot_n += 1
    tot_h += r["dur"] / 3600

# مقاطع موجودة فعلياً بلا فهرس (مثل custom غير المفهرس أعلاه) — تعدّ فقط
on_disk = sum(1 for _ in DATA_DIR.rglob("*.ogg"))

print("=" * 52)
print(f"{'المصدر':<12} {'الملفات':>9} {'الساعات':>9}")
print("-" * 52)
for k in sorted(per):
    print(f"{k:<12} {per[k][0]:>9,} {per[k][1]:>9.2f}")
print("-" * 52)
print(f"{'الإجمالي':<12} {tot_n:>9,} {tot_h:>9.2f}")
print("=" * 52)
print(f"ملفات Opus على القرص: {on_disk:,} (يشمل ما أُضيف خارج الفهرس)")
assert on_disk >= 50, f"مقاطع قليلة جداً ({on_disk}) — أعد خلية الأرشيف (7)"

# حفظ الإحصاءات مع بيانات التدريب (تُستخدم في تقرير التصدير لاحقاً)
stats = {"total_files": tot_n, "total_hours": round(tot_h, 3),
         "per_source": {k: {"files": v[0], "hours": round(v[1], 3)} for k, v in per.items()}}
(DATA_DIR / "_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n[جاهز] {tot_h:.1f} ساعة عراقية جاهزة للتدريب — تابع إلى خلية نقاط النموذج (10).")'''))

# ============================================================ CODE 10 — prefetch checkpoints
CELLS_B.append(("code", r'''# ============================================================ #
#  الخلية ١٠ — تنزيل نقاط النموذج الأساسية مسبقاً (≈2.2GB)     #
#  DiT_uvit_tat_xlsr_ema + XLSR-300m + HiFT + CAMPPlus +       #
#  OpenVoice — كلها في مجلد checkpoints داخل seed-vc           #
#  (يمنع تعطل بدء التدريب لاحقاً بسبب التنزيل، وقابل للاستئناف)#
#  عبر استيراد iraqi_pipeline داخل البيئة المعزولة — بلا CLI   #
# ============================================================ #
import subprocess, shutil
from pathlib import Path

# استدعاء cmd_prefetch من وحدة خط البيانات — داخل البيئة المعزولة (torch/huggingface_hub)
# نمرر args كما يفعل argparse تماماً (نفس الواجهة، بلا سطر أوامر)
# Q = فاصل ثلاثي-اقتباس يُبنى بـ chr(39) حتى لا يتصادم مع غلاف هذه الخلية النصية
Q = chr(39) * 3
code = (
    "import types, importlib.util as ilu, os as _os;"
    f"spec = ilu.spec_from_file_location('iraqi_pipeline', {Q}{PIPE_SRC}{Q});"
    "m = ilu.module_from_spec(spec); spec.loader.exec_module(m);"
    f"m.cmd_prefetch(types.SimpleNamespace(repo={Q}{SEEDVC}{Q}))"
)
r = subprocess.run([str(VENV_PY), "-c", code], env=ENV, text=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(r.stdout, flush=True)
assert r.returncode == 0, "فشل prefetch — راجع السجل أعلاه"

ck = SEEDVC / "checkpoints"
total = sum(f.stat().st_size for f in ck.rglob("*") if f.is_file()) / 1e9 if ck.exists() else 0
print(f"\nحجم مخزون النقاط: {total:.2f}GB في {ck}")
print("[جاهز] كل المطلوب للتدريب موجود — تابع إلى مرحلة التدريب (الخلية 11).")'''))
