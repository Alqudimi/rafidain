# ============================================================ #
#  الخلية ٢٤ — (اختيارية) تحرير مساحة القرص                    #
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
    print(f"  {t.name:<12} {s:6.2f} GB" + ("   (قابل لإعادة الجلب بالخليتين 7 و8)" if s > 0 else "   (فارغ)"))

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
    print('\nللتفعيل: اجعل  CFG["CLEAN_RAW_AFTER"] = True  في الخلية 1 ثم أعد تشغيل هذه الخلية.')
