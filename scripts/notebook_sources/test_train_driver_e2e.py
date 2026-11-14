# -*- coding: utf-8 -*-
"""In-kernel training test: run the notebook's ACTUAL cell-12 code inside a
simulated notebook kernel against a mock seed-vc repo (train.py with the same
Trainer surface) and a mock torch — proving:
  * SEED_TRAIN.Trainer is constructed with the right args (in-process, no child)
  * torch.save is wrapped BEFORE Trainer construction (mirror covers all saves)
  * stdout is teed to train_console.log
  * checkpoints land in RUN_DIR; last KEEP=2 mirrored + pruned on fake Drive
  * ft_model.pth + log mirrored at the end
  * torch.save restored after training (finally block)
  * ZERO subprocess/os.spawn usage anywhere in logic cells
"""
import json, os, re, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NB = REPO / "rafidain_colab_training.ipynb"

TMP = Path(tempfile.mkdtemp(prefix="inkernel_"))
print(f"[inkernel] workspace: {TMP}")

fails = []
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        fails.append(name)

# ---- 1) fake seed-vc repo with the real Trainer surface ----
SEEDVC = TMP / "seed-vc"
(SEEDVC / "configs" / "presets").mkdir(parents=True)
(SEEDVC / "configs" / "hifigan.yml").write_text("hift: {}\nf0_predictor: {}\n")
(SEEDVC / "configs" / "presets" / "config_dit_mel_seed_uvit_xlsr_tiny.yml").write_text(
    "log_dir: \"./runs/\"\npreprocess_params: {sr: 22050}\n")
TRAIN_PY = SEEDVC / "train.py"
TRAIN_PY.write_text(
    '''
import os
import torch

class Trainer:
    def __init__(self, config_path, pretrained_ckpt_path, data_dir, run_name,
                 batch_size=0, num_workers=0, steps=1000, save_interval=500,
                 max_epochs=1000, device="cuda:0"):
        assert os.path.exists("configs/hifigan.yml"), "kernel cwd must be SEEDVC"
        assert os.path.exists(config_path), config_path
        # torch.save must already be wrapped by the cell before construction
        assert torch.save.__name__ == "_save_with_drive_mirror", (
            f"torch.save not wrapped yet: {torch.save.__name__}")
        self.__dict__.update(locals())
    def train(self):
        os.makedirs(os.path.join("runs", self.run_name), exist_ok=True)
        for step in range(self.save_interval, self.steps + 1, self.save_interval):
            torch.save({"net": {}, "iters": step},
                       os.path.join("runs", self.run_name, f"DiT_epoch_00000_step_{step:05d}.pth"))
        torch.save({"net": {}}, os.path.join("runs", self.run_name, "ft_model.pth"))
        print(f"mock training done: {self.steps} steps, batch {self.batch_size}")
''')

# ---- 2) mock torch installed as 'torch' BEFORE the cell runs ----
MOCK_TORCH = TMP / "mock_torch.py"
MOCK_TORCH.write_text(
    '''
import types, sys, pickle
mod = types.ModuleType("torch")
def _real_save(obj, f, *a, **kw):
    with open(f, "wb") as fh:
        pickle.dump({"mock": True}, fh)
mod.save = _real_save
sys.modules["torch"] = mod
''')

# ---- 3) extract the ACTUAL cell-12 source from the built notebook ----
nb = json.loads(NB.read_text(encoding="utf-8"))
train_cell = next("".join(c["source"]) for c in nb["cells"]
                  if c["cell_type"] == "code" and "الخلية ١٢" in "".join(c["source"]))
check("cell 12 has no subprocess usage",
      not re.search(r"subprocess|os\.system|os\.spawn|Popen", train_cell))
check("cell 12 runs Trainer in-kernel", "SEED_TRAIN.Trainer(" in train_cell)

# ---- 4) simulated kernel globals (what cells 1-11 would have left behind) ----
ROOT = TMP / "iraqi_seedvc"
DATA_DIR = ROOT / "data" / "iraqi_finetune"
DATA_DIR.mkdir(parents=True)
RUN_DIR = SEEDVC / "runs" / "iraqi_v1"
DRV_RUN = TMP / "GDrive" / "runs" / "iraqi_v1"
CFG = {"RUN_NAME": "iraqi_v1", "STEPS": 30, "SAVE_EVERY": 10, "TRAIN_WORKERS": 0,
       "_BATCH": 2, "DRIVE_BACKUP": True, "DRIVE_CKPT_KEEP": 2}
SEEDVC_CFG = SEEDVC / "configs" / "presets" / "config_dit_mel_seed_uvit_xlsr_tiny.yml"

# mock SEED_TRAIN = the fake train.py loaded the same way cell 6 does
sys.path.insert(0, str(TMP))
import mock_torch                     # installs sys.modules["torch"] first
os.chdir(SEEDVC)                      # cell 6 sets kernel cwd to SEEDVC
sys.path.insert(0, str(SEEDVC))
import importlib.util as ilu
spec = ilu.spec_from_file_location("train", TRAIN_PY)
SEED_TRAIN = ilu.module_from_spec(spec)
spec.loader.exec_module(SEED_TRAIN)

G = {
    "__name__": "__main__", "ROOT": ROOT, "DATA_DIR": DATA_DIR, "RUN_DIR": RUN_DIR,
    "SEEDVC": SEEDVC, "SEEDVC_CFG": SEEDVC_CFG, "CFG": CFG, "SEED_TRAIN": SEED_TRAIN,
    "drive_run_dir": lambda: DRV_RUN,
}

# ---- 5) EXECUTE the real cell code in this kernel ----
import io, contextlib
buf = io.StringIO()
err = None
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(train_cell, "<cell-12>", "exec"), G)
except Exception as e:
    err = f"{type(e).__name__}: {e}"
out = buf.getvalue()
if err:
    print("  cell raised:", err)
    print("  stdout tail:", out[-400:])

check("cell 12 executed without error in-kernel", err is None)
check("Trainer constructed in-process", "mock training done: 30 steps, batch 2" in out)
check("drive-mirror messages printed", "[drive-mirror]" in out)
check("start banner printed (in-kernel note)", "داخل نواة الدفتر" in out)

local_ck = sorted(RUN_DIR.glob("DiT_epoch_*_step_*.pth"))
check("checkpoints saved locally (10/20/30)", len(local_ck) == 3, str([p.name for p in local_ck]))
drv_ck = sorted(DRV_RUN.glob("DiT_epoch_*_step_*.pth")) if DRV_RUN.exists() else []
check("only last 2 checkpoints on fake Drive", len(drv_ck) == 2,
      str([p.name for p in drv_ck]))
check("ft_model.pth mirrored to Drive", (DRV_RUN / "ft_model.pth").exists())
check("train_console.log written + mirrored",
      (RUN_DIR / "train_console.log").exists() and (DRV_RUN / "train_console.log").exists())

# torch.save restored to the original after the finally block
import torch as torch_mod_now
check("torch.save restored after training", torch_mod_now.save.__name__ == "_real_save")

# ---- 6) verify-cell (13) code is in-kernel too ----
verify_cell = next("".join(c["source"]) for c in nb["cells"]
                   if c["cell_type"] == "code" and "الخلية ١٣" in "".join(c["source"]))
check("cell 13 has no subprocess usage",
      not re.search(r"subprocess|os\.system|Popen", verify_cell))
check("cell 13 calls SEED_INFER.main in-kernel", "SEED_INFER.main(vc_args)" in verify_cell)

# ---- 7) prefetch/export/HF cells are direct calls ----
for no, tok in (("١٠", "PIPELINE.cmd_prefetch("), ("١٤", "PIPELINE.cmd_export(")):
    cell = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and f"الخلية {no}" in "".join(c["source"]))
    check(f"cell {no} is a direct in-kernel call", tok in cell,
          "subprocess-free: " + str(not re.search(r"subprocess|Popen", cell)))

hf_cell = next("".join(c["source"]) for c in nb["cells"]
               if c["cell_type"] == "code" and "الخلية ١٦" in "".join(c["source"]))
check("cell 16 uploads via HfApi in-kernel",
      "from huggingface_hub import HfApi" in hf_cell and "api.upload_folder(" in hf_cell
      and not re.search(r"subprocess|Popen", hf_cell))

# ---- 8) cells 4/5 are the ONLY cells allowed to mention subprocess ----
code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
offenders = []
for src in code_cells:
    if "الخلية ٤" in src or "الخلية ٥" in src:
        continue
    if re.search(r"subprocess|Popen|os\.system", src):
        offenders.append(src.splitlines()[2][:60])
check("no subprocess outside cells 4/5", not offenders, str(offenders))

print("\n" + "=" * 60)
print("IN-KERNEL TRAINING TEST:", "PASSED" if not fails else f"FAILED {fails}")
sys.exit(0 if not fails else 1)
