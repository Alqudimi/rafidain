# -*- coding: utf-8 -*-
"""
loss_monitor.py — مراقب تدريب Seed-VC في الزمن الحقيقي (عربي)
================================================================
يقرأ train_console.log (أو loss_history.jsonl) ويعرض:
  • منحنى الخسارة داخل النواة (matplotlib) + آخر القيم
  • السرعة الفعلية والإسقاط الزمني حتى الهدف
  • حكم صحة: استقرار/ميل/NaN + تقرير كامل قابل للنسخ
  • مراقبة مستمرة: loss_monitor.py --watch LOG

الاستخدام:
    python loss_monitor.py run_dir/            # قراءة loss_history.jsonl
    python loss_monitor.py train_console.log   # تحليل سجل نصي
    python loss_monitor.py LOG --watch         # تحديث كل 60 ثانية
    python loss_monitor.py LOG --target 6000 --budget-hours 11
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

STEP_RE = re.compile(r"epoch\s+(\d+)\s*,\s*step\s+(\d+)\s*,\s*loss:\s*([0-9.eE+-]+)")
TQDM_RE = re.compile(r"(\d+)/(\d+)\s*\[[^\]]*?,\s*([0-9.]+)(s/it|it/s)")
H12 = 12 * 3600


def parse_console(text):
    pts, sits, epoch_len = [], [], None
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if m:
            pts.append((int(m.group(2)), float(m.group(3))))
            continue
        m = TQDM_RE.search(line)
        if m:
            if epoch_len is None:
                epoch_len = int(m.group(2))
            v = float(m.group(3))
            sits.append(v if m.group(4) == "s/it" else 1.0 / v)
    return pts, sits, epoch_len


def parse_history(path):
    pts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
            pts.append((int(d["step"]), float(d["ema_loss"])))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return pts


def slope_per_1000(pts):
    if len(pts) < 3:
        return 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return (num / den) * 1000.0 if den else 0.0


def analyze(pts, sits, epoch_len, target, budget_hours):
    if not pts:
        return None, "لا نقاط خسارة بعد."
    vals = [v for _, v in pts]
    steps = [s for s, _ in pts]
    k = max(len(vals) // 10, 1)
    sit = statistics.median(sits) if sits else None
    r = {
        "n": len(vals), "first": steps[0], "last": steps[-1],
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals), "max": max(vals),
        "slope": slope_per_1000(pts),
        "first_k": statistics.mean(vals[:k]), "last_k": statistics.mean(vals[-k:]),
        "nan": any(v != v for v in vals),  # noqa: F632
        "sit": sit, "epoch_len": epoch_len,
    }
    lines = []
    lines.append("=" * 62)
    lines.append("مراقب التدريب — Seed-VC العراقي")
    lines.append("=" * 62)
    lines.append(f"النقاط: {r['n']} | المدى: خطوة {r['first']} → {r['last']}")
    lines.append(f"المتوسط ± الانحراف : {r['mean']:.4f} ± {r['std']:.4f}")
    lines.append(f"الأدنى / الأعلى     : {r['min']:.4f} / {r['max']:.4f}")
    lines.append(f"الميل لكل 1000 خطوة: {r['slope']:+.4f}")
    lines.append(f"أول 10% → آخر 10%  : {r['first_k']:.4f} → {r['last_k']:.4f}")
    if r["nan"]:
        lines.append("⚠ توجد قيم NaN في السجل!")
    if sit:
        lines.append(f"السرعة الوسيطة     : {sit:.2f} ث/خطوة ({1000*sit/60:.0f} دقيقة/1000 خطوة)")
        if target:
            rem = max(target - r["last"], 0)
            hrs = rem * sit / 3600
            fit = "✓ داخل جلسة 12 ساعة" if rem * sit <= H12 else "✗ تتجاوز جلسة واحدة"
            lines.append(f"حتى {target}: متبقٍ {rem} ≈ {hrs:.1f} ساعة — {fit}")
        if budget_hours:
            fit = (f"✓ يكفي ضمن ميزانية {budget_hours} ساعة"
                   if (target - r['last']) * sit <= budget_hours * 3600
                   else f"✗ لن يكتمل ضمن {budget_hours} ساعة — قلّص الهدف أو فعّل الجلسات المتعددة")
            lines.append(f"الميزانية: {fit}")
    if r["epoch_len"]:
        lines.append(f"طول الحقبة: {r['epoch_len']} دفعة ≈ {r['epoch_len']*sit/3600:.1f} ساعة" if sit else "")
    verdict = ("سليم — استمر؛ الرقم EMA منعّم 0.99 والقرار النهائي للتقييم الموضوعي"
               if not r["nan"] and r["mean"] < 1.0 and r["slope"] < 0.02
               else "يستوجب انتباهاً — راقب التقييم الموضوعي عند أقرب نقطة")
    lines.append("-" * 62)
    lines.append(f"الحكم: {verdict}")
    lines.append("=" * 62)
    return r, "\n".join(lines)


def try_plot(pts, save_to=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib غير متاح — عرض نصي فقط")
        return False
    xs = [s for s, _ in pts]
    ys = [v for _, v in pts]
    fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
    ax.plot(xs, ys, lw=1.5, color="#4C6EF5")
    mean = statistics.mean(ys)
    ax.axhline(mean, ls="--", lw=1, color="#C6866A", label=f"mean {mean:.4f}")
    ax.set_xlabel("step")
    ax.set_ylabel("EMA loss")
    ax.set_title("Seed-VC Iraqi Training")
    ax.grid(alpha=0.2)
    ax.legend()
    if save_to:
        fig.savefig(save_to, dpi=150)
        plt.close(fig)
        print(f"[plot] حُفظ المنحنى: {save_to}")
    else:
        plt.show()
    return True


def main():
    ap = argparse.ArgumentParser(description="مراقب تدريب Seed-VC (عربي)")
    ap.add_argument("src", help="ملف السجل أو مجلد التشغيل (يبحث عن loss_history.jsonl)")
    ap.add_argument("--target", type=int, default=6000)
    ap.add_argument("--budget-hours", type=float, default=0)
    ap.add_argument("--watch", action="store_true", help="مراقبة مستمرة كل 60 ثانية")
    ap.add_argument("--plot", default=None, help="حفظ المنحنى PNG إلى هذا المسار")
    args = ap.parse_args()

    def once():
        src = Path(args.src)
        if src.is_dir():
            hist = src / "loss_history.jsonl"
            console = src / "train_console.log"
            if hist.exists():
                pts = parse_history(hist)
                sits, epoch_len = [], None
            elif console.exists():
                text = console.read_text(encoding="utf-8", errors="replace")
                pts, sits, epoch_len = parse_console(text)
            else:
                print(f"لا سجل في {src} (بحثت عن loss_history.jsonl و train_console.log)")
                return None
        else:
            if src.suffix == ".jsonl":
                pts = parse_history(src)
                sits, epoch_len = [], None
            else:
                text = src.read_text(encoding="utf-8", errors="replace")
                pts, sits, epoch_len = parse_console(text)
        r, report = analyze(pts, sits, epoch_len, args.target, args.budget_hours)
        if report:
            print(report)
        if args.plot and pts:
            try_plot(pts, args.plot)
        return r

    if args.watch:
        print("[watch] مراقبة مستمرة — Ctrl+C للإيقاف")
        try:
            while True:
                print("\n" + "#" * 62)
                once()
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n[watch] أُوقفت المراقبة")
    else:
        once()


if __name__ == "__main__":
    main()
