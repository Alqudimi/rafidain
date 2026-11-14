# -*- coding: utf-8 -*-
"""End-to-end simulation of the notebook's TRAIN_DRIVER against a mock seed-vc repo.

Builds a fake seed-vc tree with a train.py that has the same import surface as the
real one (Trainer class with the same constructor signature + train() saving
checkpoints through torch.save), then runs the ACTUAL driver code extracted from
the built notebook with a mock torch module. Verifies:
  * chdir to SEEDVC happens (cwd-relative paths work)
  * Trainer constructed with the right args
  * train() runs, torch.save is wrapped BEFORE Trainer construction
  * checkpoints land in RUN_DIR and get mirrored to the fake Drive
  * ft_model.pth mirrored + train_console.log written
  * exit code 0
"""
import json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NB = REPO / "rafidain_colab_training.ipynb"

TMP = Path(tempfile.mkdtemp(prefix="driver_e2e_"))
print(f"[e2e] workspace: {TMP}")

# ---- 1) fake seed-vc repo ----
SEEDVC = TMP / "seed-vc"
(SEEDVC / "configs" / "presets").mkdir(parents=True)
(SEEDVC / "configs").mkdir(exist_ok=True)
# the real train.py opens configs/hifigan.yml relative to cwd at build_vocoder time
(SEEDVC / "configs" / "hifigan.yml").write_text("hift: {}\nf0_predictor: {}\n")
(SEEDVC / "configs" / "presets" / "config_dit_mel_seed_uvit_xlsr_tiny.yml").write_text(
    "log_dir: \"./runs/\"\npreprocess_params: {sr: 22050}\n")
# mock train.py — same constructor signature as the real one
(SEEDVC / "train.py").write_text(
    '''
import os, sys, json, torch

class Trainer:
    def __init__(self, config_path, pretrained_ckpt_path, data_dir, run_name,
                 batch_size=0, num_workers=0, steps=1000, save_interval=500,
                 max_epochs=1000, device="cuda:0"):
        self.config_path = config_path
        self.pretrained = pretrained_ckpt_path
        self.data_dir = data_dir
        self.run_name = run_name
        self.batch_size = batch_size
        self.steps = steps
        self.save_interval = save_interval
        self.device = device
        # prove cwd was set to SEEDVC by the driver
        assert os.path.exists("configs/hifigan.yml"), "driver must chdir to SEEDVC"
        # prove torch.save was wrapped BEFORE Trainer construction
        assert torch.save.__name__ == "_save_with_drive_mirror", (
            f"torch.save not wrapped yet: {torch.save.__name__}")
        # the config must exist (driver passed the right path)
        assert os.path.exists(config_path), config_path

    def train(self):
        os.makedirs(os.path.join("runs", self.run_name), exist_ok=True)
        for step in range(self.save_interval, self.steps + 1, self.save_interval):
            state = {"net": {}, "iters": step}
            torch.save(state, os.path.join("runs", self.run_name,
                        f"DiT_epoch_00000_step_{step:05d}.pth"))
        torch.save({"net": {}}, os.path.join("runs", self.run_name, "ft_model.pth"))
        print(f"mock training done: {self.steps} steps, batch {self.batch_size}")
''')
# a minimal torch shim that records saves and forwards to the driver's wrapper
# (the driver imports torch first, wraps save; train.py's `import torch` gets the same module)
MOCK_TORCH = TMP / "mock_torch.py"
MOCK_TORCH.write_text(
    '''
import types, sys, pickle, os
mod = types.ModuleType("torch")
_real_dir = os.environ["MOCK_TORCH_DIR"]
def _real_save(obj, f, *a, **kw):
    # emulate torch.save: pickle to file
    import pickle
    with open(f, "wb") as fh:
        pickle.dump({"mock": True, "obj": str(obj)}, fh)
mod.save = _real_save
sys.modules["torch"] = mod
''')

# ---- 2) extract TRAIN_DRIVER from the built notebook and write driver file ----
nb = json.loads(NB.read_text(encoding="utf-8"))
train_cell = next("".join(c["source"]) for c in nb["cells"]
                  if c["cell_type"] == "code" and "الخلية ١٢" in "".join(c["source"]))
m = re.search(r"TRAIN_DRIVER = \((.*?)\n    \)\n", train_cell, re.S)
assert m, "TRAIN_DRIVER not found"
scope = {}
exec("TQ = chr(34) * 3\n" + m.group(0), scope)
driver_src = scope["TRAIN_DRIVER"]
DRIVER = TMP / "train_driver.py"
DRIVER.write_text(driver_src, encoding="utf-8")

# ---- 3) run driver with system python + mock torch preload ----
RUN_DIR = SEEDVC / "runs" / "iraqi_v1"
DATA_DIR = TMP / "data"
DATA_DIR.mkdir()
CFG_PATH = TMP / "cfg.json"
CFG_PATH.write_text(json.dumps({"RUN_NAME": "iraqi_v1"}))
DRV_RUN = TMP / "GDrive" / "runs" / "iraqi_v1"

loader = f'''
import sys, types
sys.path.insert(0, r"{TMP}")
import mock_torch  # installs sys.modules["torch"] BEFORE the driver imports it
sys.argv = ["train_driver.py", r"{SEEDVC}", r"{RUN_DIR}", r"{DATA_DIR}", r"{CFG_PATH}",
            r"{DRV_RUN}", "2", "2", "30", "10", "0"]
exec(open(r"{DRIVER}", encoding="utf-8").read())
'''
LOADER = TMP / "loader.py"
LOADER.write_text(loader, encoding="utf-8")

r = subprocess.run([sys.executable, str(LOADER)], capture_output=True, text=True,
                   env={**os.environ, "MOCK_TORCH_DIR": str(TMP)})
print("--- driver stdout/stderr ---")
print(r.stdout[-1500:])
if r.stderr:
    print("STDERR:", r.stderr[-800:])
print("--- exit code:", r.returncode, "---")

ok = True
def check(name, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        ok = False

check("driver exited 0", r.returncode == 0)
check("training console log written", (RUN_DIR / "train_console.log").exists())
local_ck = sorted(RUN_DIR.glob("DiT_epoch_*_step_*.pth"))
check("checkpoints saved locally (step 10,20,30)", len(local_ck) == 3, str([p.name for p in local_ck]))
drv_ck = sorted(DRV_RUN.glob("DiT_epoch_*_step_*.pth")) if DRV_RUN.exists() else []
check("only last 2 checkpoints mirrored to Drive", len(drv_ck) == 2,
      str([p.name for p in drv_ck]))
check("ft_model.pth mirrored to Drive", (DRV_RUN / "ft_model.pth").exists())
check("log mirrored to Drive", (DRV_RUN / "train_console.log").exists())
check("driver printed drive-mirror messages", "[drive-mirror]" in r.stdout)
check("driver printed start banner with steps/keep", "[train] بدء الضبط الدقيق" in r.stdout)

print("\n" + "=" * 60)
print("TRAIN DRIVER E2E:", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
