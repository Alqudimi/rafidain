# ============================================================ #
#  الخلية ٩ — (اختيارية) دمج تسجيلاتك الخاصة في التدريب         #
#  ارفع ملفاتك (wav/flac/mp3/m4a/opus/ogg) إلى:                #
#     /workspace/iraqi_seedvc/raw/custom/                      #
#  عبر متصفح ملفات Jupyter (سحب وإفلات) ثم أعد تشغيل الخلية    #
#  ستُقطَّع تلقائياً إلى مقاطع 1-29.5 ثانية وتُرمَّز Opus 24kHz   #
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
  1) ارفع تسجيلاتك إلى المجلد أعلاه (سحب وإفلات في لوحة ملفات Jupyter)
  2) الأفضل: كلام نظيف بلا موسيقى أو ضجيج، 1-30 ثانية أو أطول (تُقطَّع تلقائياً)
  3) أعد تشغيل هذه الخلية، ثم شغّل خلية التقطيع (10) لتشمل بياناتك
(تخطَّ هذه الخلية إن لم يكن لديك بيانات خاصة — مجموعة YODAS + الصغيرة كافية تماماً)''')
else:
    sys.path.insert(0, str(ROOT))
    import iraqi_pipeline as P   # خط البيانات المستخرج في الخلية 6 — نفس منطق التقطيع المجرَّب
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
    print("فهرس بيانات التدريب محدّث — أكمل إلى الخلية 10.")
