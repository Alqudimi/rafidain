# ============================================================ #
#  الخلية ١١ — تقرير بيانات التدريب (جديد في الإصدار ٢)        #
#  ساعات كل مصدر + مدرّج أطوال المقاطع + فحص جودة سريع        #
#  (تسميات الرسم بالإنجليزية لأن matplotlib لا يشكّل            #
#   الحروف العربية المتصلة بشكل صحيح)                          #
# ============================================================ #
import json
from pathlib import Path

# ---- إحصاء من فهرس التجهيز + من الملفات الفعلية ----
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

per, tot_h, tot_n = {}, 0.0, 0
for r in rows:
    src = r["file"].split("/")[0] if "/" in r["file"] else "(جذر)"
    per.setdefault(src, [0, 0.0])
    per[src][0] += 1
    per[src][1] += float(r["dur"]) / 3600
    tot_n += 1
    tot_h += float(r["dur"]) / 3600

print("=" * 52)
print("تقرير بيانات التدريب المجهزة")
print("=" * 52)
for k in sorted(per):
    print(f"  {k:<10}: {per[k][0]:6,} مقاطع | {per[k][1]:7.2f} ساعة")
print(f"  {'الإجمالي':<10}: {tot_n:6,} مقاطع | {tot_h:7.2f} ساعة")
print("=" * 52)

if tot_h < 1:
    print("[تحذير] أقل من ساعة واحدة — التدريب سيكون ضعيفاً. زد YODAS_HOURS أو أضف بياناتك.")
elif tot_h < 10:
    print("[ملاحظة] أقل من 10 ساعات — يكفي لتجربة المسار، ولجودة إنتاجية زد البيانات.")
else:
    print(f"[جيد] {tot_h:.0f} ساعة عراقية — جرعة تدريب قوية.")

# ---- مدرّج أطوال المقاطع ----
if rows:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    durs = [float(r["dur"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
    ax.hist(durs, bins=60, color="#1f77b4", edgecolor="white", linewidth=0.4)
    ax.axvline(sum(durs) / len(durs), color="#d62728", lw=1.6,
               label=f"mean = {sum(durs)/len(durs):.1f}s")
    ax.set_xlabel("segment duration (s)")
    ax.set_ylabel("segments")
    ax.set_title(f"Iraqi training segments - {tot_n:,} files / {tot_h:.1f} h")
    ax.legend()
    ax.grid(alpha=0.3)
    chart = ROOT / "data" / "duration_hist.png"
    fig.savefig(chart, dpi=110)
    plt.close(fig)
    from IPython.display import display, Image as IPyImage
    display(IPyImage(filename=str(chart)))
    print(f"(حُفظ بشكل دائم: {chart} — سيُرفق بحزمة النموذج)")
    # فحص جودة: نسبة المقاطع خارج نطاق 1.5-29.5 ث
    bad = sum(1 for d in durs if d < 1.5 or d > 29.9)
    print(f"مقاطع خارج النطاق المثالي: {bad} ({100*bad/len(durs):.1f}%) — ستُتخطى تلقائياً في التدريب")
