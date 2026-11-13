# ============================================================ #
#  الخلية ٤ — بيئة Python المعزولة + PyTorch 2.4.0 (CUDA 12.1) #
#  نفس الإصدارات المثبتة في سكربتات المستودع الرسمية            #
#  (تُبنى مرة واحدة على القرص الدائم وتُعاد استخدامها دائماً)    #
# ============================================================ #
import subprocess, shutil, sys
from pathlib import Path

UV = globals().get("UV") or shutil.which("uv")   # تحصين: يعمل حتى بعد إعادة تشغيل النواة

def venv_ok(code):
    # فحص سريع داخل البيئة المعزولة
    if not VENV_PY.exists():
        return False
    r = subprocess.run([str(VENV_PY), "-c", code], capture_output=True, text=True, env=ENV)
    return r.returncode == 0

# ---- 1) إنشاء venv إن لم يوجد ----
if not VENV_PY.exists():
    print(f"إنشاء بيئة معزولة (Python 3.11) في {ROOT/'venv'} ...")
    if UV:
        sh([UV, "venv", "--python", "3.11", str(ROOT / "venv")])
    else:
        sh([sys.executable, "-m", "venv", str(ROOT / "venv")])

# ---- 2) PyTorch 2.4.0 cu121 (نحو 2.5GB — يُتخطى إن كان مثبتاً) ----
if not venv_ok("import torch; assert torch.__version__.startswith('2.4.')"):
    print("تثبيت PyTorch 2.4.0 + torchaudio (cu121) — عدة دقائق حسب الشبكة ...")
    trio_pin = ["torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0"]
    trio_any = ["torch", "torchvision", "torchaudio"]
    idx = ["--index-url", "https://download.pytorch.org/whl/cu121"]
    if UV:
        rc = sh([UV, "pip", "install", "--python", str(VENV_PY), *trio_pin, *idx], check=False)
        if rc != 0:
            sh([UV, "pip", "install", "--python", str(VENV_PY), *trio_any, *idx])
    else:
        rc = sh([str(VENV_PY), "-m", "pip", "install", "-q", *trio_pin, *idx], check=False)
        if rc != 0:
            sh([str(VENV_PY), "-m", "pip", "install", "-q", *trio_any, *idx])
else:
    print("PyTorch 2.4.x موجود أصلاً في البيئة — تخطّي التثبيت.")

# ---- 3) بقية الاعتماديات (مثبتة بالإصدارات نفسها التي نجح بها خط المستودع) ----
NEED = ("import torch, torchaudio, librosa, soundfile, munch, einops,"
        " transformers, pyarrow, huggingface_hub, yaml, matplotlib")
if not venv_ok(NEED):
    print("تثبيت الاعتماديات (numpy/scipy/librosa/transformers/...) ...")
    pins = ["numpy==1.26.4", "scipy==1.13.1", "librosa==0.10.2", "soundfile==0.12.1",
            "munch==4.0.0", "einops==0.8.0", "transformers==4.46.3", "pyyaml", "tqdm",
            "requests", "pyarrow==17.0.0", "huggingface_hub", "matplotlib"]
    loose = ["numpy", "scipy", "librosa", "soundfile", "munch", "einops", "transformers",
             "pyyaml", "tqdm", "requests", "pyarrow", "huggingface_hub", "matplotlib"]
    if UV:
        rc = sh([UV, "pip", "install", "--python", str(VENV_PY), *pins], check=False)
        if rc != 0:
            sh([UV, "pip", "install", "--python", str(VENV_PY), *loose])
    else:
        rc = sh([str(VENV_PY), "-m", "pip", "install", "-q", *pins], check=False)
        if rc != 0:
            sh([str(VENV_PY), "-m", "pip", "install", "-q", *loose])
else:
    print("الاعتماديات موجودة أصلاً — تخطّي التثبيت.")

# ---- 4) تحقق نهائي (بما فيه رؤية GPU من داخل البيئة) ----
r = subprocess.run([str(VENV_PY), "-c",
                    "import torch, torchaudio; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"],
                   capture_output=True, text=True, env=ENV)
print((r.stdout + r.stderr).strip())
assert r.returncode == 0 and "torch 2.4" in (r.stdout + r.stderr), "فشل بناء البيئة — راجع السجل أعلاه"
print(f"\n[جاهز] البيئة المعزولة: {VENV_PY}")
