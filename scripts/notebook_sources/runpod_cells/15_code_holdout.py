# ============================================================ #
#  الخلية ١٢ — عزل بيانات الاختبار (جديد في الإصدار ٢)         #
#  نستبعد عدداً من المقاطع العراقية من التدريب نهائياً،         #
#  لتقييم النموذج لاحقاً على كلام لم تره عيناه إطلاقاً          #
#  (التقييم العلمي الوحيد المعتدل — وإلا فأنت تختبر الحفظ)     #
# ============================================================ #
import json, random
from pathlib import Path

TEST_DIR.mkdir(parents=True, exist_ok=True)
hl_path = ROOT / "data" / "holdout_list.json"

# 1) اختيار حتمي (نفس البذرة دائماً = نفس القائمة بين الجلسات)
idx = DATA_DIR / "_prep_index.jsonl"
rows = []
if idx.exists():
    for line in idx.read_text().splitlines():
        try:
            r = json.loads(line)
            if (DATA_DIR / r["file"]).exists():
                rows.append(r)
        except Exception:
            pass
assert rows, "لا يوجد فهرس بيانات — شغّل الخلية 10 (التقطيع) أولاً"

if hl_path.exists():
    holdout = json.loads(hl_path.read_text())
    print(f"قائمة اختبار موجودة: {len(holdout)} مقطعاً (من اختيار سابق — تُحترم كما هي)")
else:
    rng = random.Random(42)   # بذرة ثابتة: قابلية إعادة إنتاج كاملة
    # نفضل مقاطع 3-15 ثانية (مثالية كعينات تقييم: ليست قصيرة جداً ولا ضخمة)
    pool = [r for r in rows if 3.0 <= float(r["dur"]) <= 15.0]
    if len(pool) < CFG["HOLDOUT_COUNT"]:
        pool = rows
    take = min(int(CFG["HOLDOUT_COUNT"]), len(pool))
    # تنويع المصادر: نوزع الاختيار بالتساوي على المصادر المتاحة
    by_src = {}
    for r in pool:
        by_src.setdefault(r["file"].split("/")[0] if "/" in r["file"] else "root", []).append(r)
    for lst in by_src.values():
        rng.shuffle(lst)
    holdout, i = [], 0
    while len(holdout) < take:
        added = False
        for lst in by_src.values():
            if i < len(lst) and len(holdout) < take:
                holdout.append(lst[i]["file"])
                added = True
        if not added:
            break
        i += 1
    hl_path.write_text(json.dumps(holdout, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"قائمة اختبار جديدة: {len(holdout)} مقطعاً (بذرة ثابتة 42 — موزعة على المصادر)")

# 2) تنفيذ العزل الفعلي (idempotent: يعمل بأمان أي عدد من المرات)
moved = kept = 0
for rel in holdout:
    src = DATA_DIR / rel
    dst = TEST_DIR / Path(rel).name
    if src.exists():
        if dst.exists():
            src.unlink()          # موجود في جهة الاختبار أصلاً — نحذف نسخة التدريب
        else:
            src.rename(dst)
        moved += 1
    elif dst.exists():
        kept += 1

print(f"عُزل الآن: {moved} | كان معزولاً سابقاً: {kept} | إجمالي الاختبار: {len(list(TEST_DIR.glob('*.ogg')))} مقطعاً")
print(f"مجلد الاختبار: {TEST_DIR}")

# 3) تحديث إحصاءات التدريب بعد الاستبعاد (يحذف من الفهرس المداخل غير الموجودة)
sh([VENV_PY, PIPE, "stats", "--dir", DATA_DIR], check=False)

# ملاحظة: إذا أعدت تشغيل الخلية 10 (prep) لاحقاً فقد تعود بعض المقاطع لمجلد التدريب —
# لا مشكلة: خلية إطلاق التدريب (14) تعيد العزل تلقائياً قبل الانطلاق (reapply_holdout).
print("\n[جاهز] تابع إلى الخلية 13 (تنزيل نقاط النموذج).")
