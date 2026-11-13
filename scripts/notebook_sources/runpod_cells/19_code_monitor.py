# ============================================================ #
#  الخلية ١٥ — مراقبة التدريب (أعد تشغيلها متى شئت)            #
#  تقدم + معدل + زمن متبقٍ + منحنى الخسارة + آخر السجل          #
#  + حارس NaN: يكشف انهيار الخسارة فوراً (جديد في الإصدار ٢)    #
# ============================================================ #
import re, os, time
from pathlib import Path
from IPython.display import display, Image as IPyImage

LOG = RUN_DIR / "train_console.log"
PIDF = RUN_DIR / "train.pid"
FT = RUN_DIR / "ft_model.pth"
STEP_RE = re.compile(r"epoch (\d+), step (\d+), loss: ([0-9.eEnaif+-]+)")
MAXS = int(CFG["STEPS"])

def _alive():
    try:
        if PIDF.exists():
            os.kill(int(PIDF.read_text().strip()), 0)
            return True
    except Exception:
        pass
    return False

lines = LOG.read_text(errors="replace").splitlines() if LOG.exists() else []
pts = []
for ln in lines:
    m = STEP_RE.search(ln)
    if m:
        try:
            pts.append((int(m.group(2)), float(m.group(3))))
        except ValueError:
            pass

if FT.exists():
    status = "منجز — ft_model.pth موجود (انتقل إلى خلية التحقق 17)"
elif _alive():
    status = "يعمل الآن"
elif pts:
    status = "متوقف — العملية غير حية ولا يوجد نموذج نهائي (راجع آخر السجل)"
else:
    status = "لم يبدأ بعد"

steps = pts[-1][0] if pts else 0
start = PIDF.stat().st_mtime if PIDF.exists() else (LOG.stat().st_mtime if LOG.exists() else time.time())
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

# ---- حارس NaN / Inf (انهيار التدريب المبكر أشهر فشل صامت) ----
if pts and any((p[1] != p[1]) or float("inf") == abs(p[1]) for p in pts[-20:]):
    print("\n" + "!" * 62)
    print("تحذير: الخسارة انهارت إلى NaN/Inf!")
    print("الإجراء: أوقف التدريب (احذف train.pid وأعد الخلية 14) — سيبدأ من آخر نقطة سليمة.")
    print("إن تكرر: خفّض BATCH إلى النصف في الخلية 1.")
    print("!" * 62)

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
