# ============================================================ #
#  الخلية ٢ — فحص البيئة: GPU / قرص / أدوات                    #
#  (يحدد حجم الدفعة تلقائياً ويكشف مدى ديمومة التخزين)          #
# ============================================================ #
import subprocess, os, shutil, re, json
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
else:
    print("[GPU] لا يوجد GPU ظاهر عبر nvidia-smi — لن يعمل التدريب (تجهيز البيانات فقط).")

def auto_batch(vram_gb):
    # دفعة آمنة لنموذج tiny (25M) مع مكوناته الثقيلة (XLSR-300m + HiFT + OpenVoice)
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
    print(f"[نظام] {cpu} نواة معالج — ذاكرة RAM ≈ {mem_kb/1e6:.1f}GB")
except Exception:
    print(f"[نظام] {cpu} نواة معالج")

# ---- القرص ونوع التخزين (ديمومة الملفات!) ----
du = shutil.disk_usage(ROOT)
print(f"[قرص] المتاح على {ROOT}: حر {du.free/1e9:.1f}GB من {du.total/1e9:.1f}GB")
def find_mount(path):
    best, fstype, dev = "", "", ""
    try:
        for line in Path("/proc/mounts").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3 and str(path).startswith(parts[1]) and len(parts[1]) > len(best):
                best, fstype, dev = parts[1], parts[2], parts[0]
    except Exception:
        pass
    return best, fstype, dev
mnt, fstype, dev = find_mount(ROOT)
print(f"[تخزين] نقطة الوصل: {mnt or '/'} — النوع: {fstype or 'غير معروف'} — الجهاز: {dev or '?'}")
if str(ROOT).startswith("/workspace"):
    print("        [جيد] العمل تحت /workspace — إن كان هذا القرص الشبكي (Network Volume) فملفاتك")
    print("               دائمة وتبقى بعد إيقاف/حذف الحاوية (لا تحذف الـ Volume من لوحة RunPod).")
elif str(ROOT).startswith("/root") or str(ROOT).startswith("/home"):
    print("        [تحذير] لا يوجد /workspace — أنت على قرص الحاوية المحلي:")
    print("                يبقى عند الإيقاف (Stop) لكنه يضيع عند الحذف (Terminate).")
    print("                لضمان الديمومة: أنشئ Network Volume في RunPod وأعد تشغيل الدفتر،")
    print("                أو اضبط IRAQI_DATA_ROOT، أو اعتمد على الرفع الدائم (الخلية 22).")

# ---- تقدير مساحة التنزيل المطلوبة ----
need_gb = 2.2 + 6.0   # نقاط النماذج + بيئة Python/PyTorch
t = str(CFG["YODAS_HOURS"]).strip().lower()
hours_label = "0"
if t not in ("0", ""):
    hours_label = "كل الكتالوج" if t in ("all", "كل") else f"{float(t):.0f} ساعة"
    # الخام ~ 3GB لكل 20 ساعة تقريباً (WebM) + المقاطع المشفرة Opus ~ 0.15GB/ساعة
    est_raw = 3.0 * (339 if t in ("all", "كل") else float(t)) / 20.0
    need_gb += est_raw + est_raw * 0.15
print(f"[تقدير] يحتاج هذا التشغيل ({hours_label} يوداس + مجموعات صغيرة) ≈ {need_gb:.0f}GB تقريباً")
if du.free / 1e9 < need_gb:
    print(f"        [تحذير] المساحة الحرة ({du.free/1e9:.1f}GB) قد لا تكفي!")
    print("                خفّض YODAS_HOURS، أو وسّع القرص الشبكي، أو فعّل CLEAN_RAW_AFTER (الخلية 24).")
else:
    print("        [جيد] المساحة الحرة كافية.")

# ---- الأدوات ----
for tool in ("ffmpeg", "ffprobe", "git", "curl", "uv"):
    print(f"[أداة] {tool:<8}: {'متوفر' if shutil.which(tool) else 'غير متوفر'}")

print("\n[فحص مبدئي] أكمل إلى الخلية 3 (التبعيات).")
