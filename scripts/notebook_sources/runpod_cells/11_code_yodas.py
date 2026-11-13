# ============================================================ #
#  الخلية ٨ — جلب بيانات YODAS العراقية (الكتلة الكبرى)        #
#  تنزيل مُجزَّأ متوازٍ من أرشيفات espnet/yodas3                #
#  يستغرق عادة 15-60 دقيقة (حسب الشبكة وساعاتك)                #
#  قابلاً للاستئناف 100%: أعد التشغيل ليكمل من حيث توقف         #
# ============================================================ #
import shutil

t = str(CFG["YODAS_HOURS"]).strip().lower()
if t == "0":
    print("YODAS معطّل (YODAS_HOURS=0) — تخطّي.")
else:
    free0 = shutil.disk_usage(ROOT).free / 1e9
    print(f"المساحة الحرة قبل الجلب: {free0:.1f}GB")
    sh([VENV_PY, PIPE, "yodas",
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
print("\n[جاهز] تابع إلى الخلية 9 (اختيارية) ثم 10 (التقطيع والترميز).")
