# ============================================================ #
#  الخلية ٢٠ — تصدير الحزمة النهائية                           #
#  أوزان + إعداد + تقارير تقييم + بصمات إعادة الإنتاج          #
#  في مجلد ثابت على القرص الدائم: exports/<اسم-التشغيل>/       #
# ============================================================ #
import shutil, json
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

# ---- توثيق إضافي يُنسخ مع الحزمة (قابلية إعادة الإنتاج جزء من الجودة) ----
extras = [
    RUN_DIR / "train_console.log",
    RUN_DIR / "loss_curve.png",
    RUN_DIR / "config_dit_mel_seed_uvit_xlsr_tiny.yml",
    DATA_DIR / "_stats.json",
    ROOT / "data" / "duration_hist.png",        # مدرّج البيانات (جديد)
    ROOT / "repro.json",                         # بصمات git (جديد)
    ROOT / "nb_config.json",                     # مقابض التشغيل (جديد)
]
for extra in extras:
    if extra.exists():
        shutil.copy2(extra, EXPORT_DIR / extra.name)
        print(f"  + {extra.name}")

# ---- تضمين تقارير التقييم والبنشمارك داخل الحزمة (جديد) ----
for sub, dst_name in ((RUN_DIR / "ab_eval", "ab_eval"),
                      (RUN_DIR / "bench", "bench_stream")):
    if sub.exists():
        (EXPORT_DIR / dst_name).mkdir(exist_ok=True)
        for f in sub.iterdir():
            if f.is_file():
                shutil.copy2(f, EXPORT_DIR / dst_name / f.name)
        print(f"  + {dst_name}/ ({len(list((EXPORT_DIR / dst_name).iterdir()))} ملفات)")

# ---- دمج كل التقارير في ملف training_report.json واحد غني ----
report_path = EXPORT_DIR / "training_report.json"
report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
if (ROOT / "repro.json").exists():
    report["reproducibility"] = json.loads((ROOT / "repro.json").read_text(encoding="utf-8"))
ab_json = EXPORT_DIR / "ab_eval" / "ab_eval.json"
if ab_json.exists():
    report["ab_evaluation"] = json.loads(ab_json.read_text(encoding="utf-8"))["summary"]
    report["ab_evaluation"]["note"] = ("CAMPPlus speaker similarity + XLSR content "
                                       "preservation, held-out Iraqi pairs, identical seeds")
bench_json = EXPORT_DIR / "bench_stream" / "bench_stream.json"
if bench_json.exists():
    report["latency_benchmark"] = json.loads(bench_json.read_text(encoding="utf-8"))
report["training_config"] = {k: CFG[k] for k in CFG if not k.startswith("_") and k != "HF_TOKEN"}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("  + training_report.json (محدّث بالتقييم والبنشمارك والبصمات)")

total = 0
print(f"\nحزمة التصدير: {EXPORT_DIR}")
for p in sorted(EXPORT_DIR.rglob("*")):
    if p.is_file():
        s = p.stat().st_size
        total += s
        rel = p.relative_to(EXPORT_DIR)
        print(f"  {str(rel):<46} {s/1e6:9.2f} MB")
print(f"  {'الإجمالي':<46} {total/1e6:9.2f} MB")
