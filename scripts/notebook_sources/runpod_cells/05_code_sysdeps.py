# ============================================================ #
#  الخلية ٣ — تبعيات النظام (ffmpeg / git) + مدير الحزم uv     #
#  (حاويات RunPod تعمل بصلاحيات root عادة — بلا sudo)          #
# ============================================================ #
import shutil, os, subprocess
from pathlib import Path

def have(x):
    return shutil.which(x) is not None

missing = [p for p in ("ffmpeg", "git") if not have(p)]
if missing:
    print(f"تثبيت حزم النظام: {', '.join(missing)} ...")
    sh(["apt-get", "update", "-qq"], check=False)
    sh(["apt-get", "install", "-y", "-qq", *missing])
for tool in ("ffmpeg", "git", "curl"):
    print(f"  {tool:<8}: {'متوفر' if have(tool) else 'غير متوفر!'}")

# ---- uv: مُسرّع تثبيت بايثون (نفس مسار سكربتات المستودع) ----
UV = shutil.which("uv")
if UV is None:
    print("\nتثبيت uv (مرة واحدة) ...")
    sh(["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], check=False)
    for cand in (Path.home() / ".local" / "bin" / "uv", Path.home() / ".cargo" / "bin" / "uv"):
        if cand.exists():
            UV = str(cand)
            break
    if UV:
        os.environ["PATH"] = str(Path(UV).parent) + os.pathsep + os.environ["PATH"]
        ENV["PATH"] = os.environ["PATH"]
if UV:
    r = subprocess.run([UV, "--version"], capture_output=True, text=True)
    print(f"\nuv جاهز: {r.stdout.strip() or r.stderr.strip()}")
else:
    print("\nuv غير متوفر — سنعتمد python3 -m venv + pip كبديل في الخلية 4 (يعمل أيضاً).")
