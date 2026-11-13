# ============================================================ #
#  الخلية ١٦ — (اختيارية) انتظار اكتمال التدريب                #
#  مراقبة حية تلقائية كل 30 ثانية حتى النهاية                  #
#  (بدلها: شغّل الخلية 15 يدوياً كل فترة — نفس الغرض)           #
# ============================================================ #
import re, os, time
from pathlib import Path
from IPython.display import clear_output

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

deadline = time.time() + 24 * 3600   # مهلة قصوى 24 ساعة
while time.time() < deadline:
    clear_output(wait=True)
    lines = LOG.read_text(errors="replace").splitlines() if LOG.exists() else []
    steps, loss = 0, None
    for ln in lines:
        m = STEP_RE.search(ln)
        if m:
            try:
                steps, loss = int(m.group(2)), float(m.group(3))
            except ValueError:
                pass
    pct = min(100.0 * steps / MAXS, 100.0)
    bar = "#" * int(30 * pct / 100)
    msg = f"[{bar:<30}] {pct:5.1f}%   step {steps}/{MAXS}"
    if loss is not None:
        msg += f"   loss {loss:.4f}"
        if loss != loss or loss == float("inf"):
            msg += "   << NaN/Inf — أوقف التدريب واستأنف من آخر نقطة!"
    print(msg, flush=True)
    if FT.exists():
        print("\nاكتمل التدريب — ft_model.pth جاهز. انتقل إلى خلية التحقق (17).")
        break
    if not _alive():
        print("\nتوقفت عملية التدريب دون إنتاج النموذج النهائي.")
        print("شغّل خلية المراقبة (15) لرؤية آخر السجل — غالباً يُستأنف بإعادة الخلية 14.")
        break
    time.sleep(30)
else:
    print("انتهت مهلة الانتظار (24 ساعة) — استخدم الخلية 15 للفحص اليدوي.")
