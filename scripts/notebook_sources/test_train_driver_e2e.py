# -*- coding: utf-8 -*-
"""Run Colab's current in-kernel training cell against a deterministic fake trainer."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
NB = REPO / "rafidain_colab_training.ipynb"
TMP = Path(tempfile.mkdtemp(prefix="inkernel_"))
print(f"[inkernel] workspace: {TMP}")

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def code_cell(number: str) -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if f"الخلية {number}" in source:
                return source
    raise RuntimeError(f"notebook code cell {number} not found")


def code_cell_with(token: str) -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if token in source:
                return source
    raise RuntimeError(f"notebook code cell containing {token!r} not found")


# Mock Seed-VC repository. Preserve the exact upstream patch anchors as inert
# string constants, while Trainer below captures the public in-kernel API.
SEEDVC = TMP / "seed-vc"
(SEEDVC / "configs" / "presets").mkdir(parents=True)
(SEEDVC / "configs" / "hifigan.yml").write_text("hift: {}\nf0_predictor: {}\n")
CFG_MODEL = SEEDVC / "configs" / "presets" / "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
CFG_MODEL.write_text("log_dir: ./runs/\n")
from finetune.upstream_patch import _OLD_LOAD, _OLD_TRAIN

TRAIN_PY = SEEDVC / "train.py"
TRAIN_PY.write_text(
    "import os\n"
    "class _PatchAnchors:\n"
    "    load = r'''\n" + _OLD_LOAD + "\n'''\n"
    "    train = r'''\n"
    + _OLD_TRAIN
    + "\n'''\n"
    + """
import os
import torch

class Trainer:
    def __init__(self, config_path, pretrained_ckpt_path, data_dir, run_name,
                 batch_size=0, num_workers=0, steps=1000, save_interval=500,
                 max_epochs=1000, device="cuda:0"):
        assert os.path.exists("configs/hifigan.yml"), "kernel cwd must be SEEDVC"
        assert os.path.exists(config_path), config_path
        self.__dict__.update(locals())

    def train(self):
        run_dir = os.path.join("runs", self.run_name)
        os.makedirs(run_dir, exist_ok=True)
        for step in range(self.save_interval, self.steps + 1, self.save_interval):
            torch.save({"net": {}, "optimizer": {}, "iters": step},
                       os.path.join(run_dir, f"DiT_epoch_00000_step_{step:05d}.pth"))
        torch.save({"net": {}}, os.path.join(run_dir, "ft_model.pth"))
        print(f"mock training done: {self.steps} steps, batch {self.batch_size}")
"""
)

# Install a tiny torch substitute before the actual notebook cell imports it.
MOCK_TORCH = TMP / "mock_torch.py"
MOCK_TORCH.write_text(
    """
import types, sys, pickle
mod = types.ModuleType("torch")
def _real_save(obj, f, *a, **kw):
    with open(f, "wb") as fh:
        pickle.dump({"mock": True}, fh)
mod.save = _real_save
sys.modules["torch"] = mod
"""
)

train_cell = code_cell("١٤")
check("training cell invokes Trainer directly", "SEED_TRAIN.Trainer(" in train_cell)
check(
    "training cell does not launch a child training process",
    "subprocess.Popen" not in train_cell and "start_new_session" not in train_cell,
)
check(
    "training cell installs guarded full-state resume patch",
    "ensure_full_state_resume(SEEDVC)" in train_cell,
)
check(
    "embedded Drive mirror promotes only stable copies",
    "def _copy_atomic(src, dst):" in train_cell
    and "os.replace(tmp, dst)" in train_cell
    and "source changed during checkpoint copy" in train_cell,
)

ROOT = TMP / "iraqi_seedvc"
ROOT.mkdir()
DATA_DIR = ROOT / "data" / "iraqi_finetune"
DATA_DIR.mkdir(parents=True)
RUN_DIR = SEEDVC / "runs" / "iraqi_v1"
DRIVE_ROOT = TMP / "GDriveRoot"
DRIVE_ROOT.mkdir()
CFG = {
    "RUN_NAME": "iraqi_v1",
    "STEPS": 30,
    "SAVE_EVERY": 10,
    "TRAIN_WORKERS": 0,
    "_BATCH": 2,
    "DRIVE_BACKUP": True,
    "DRIVE_CKPT_KEEP": 2,
}
ENV = dict(os.environ)
RAFIDAIN = REPO
DRIVE_OK = True

sys.path.insert(0, str(TMP))
import mock_torch  # noqa: F401

os.chdir(SEEDVC)
sys.path.insert(0, str(SEEDVC))
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("train", TRAIN_PY)
SEED_TRAIN = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(SEED_TRAIN)

G = {
    "__name__": "__main__",
    "ROOT": ROOT,
    "DATA_DIR": DATA_DIR,
    "RUN_DIR": RUN_DIR,
    "SEEDVC": SEEDVC,
    "RAFIDAIN": RAFIDAIN,
    "CFG_MODEL": CFG_MODEL,
    "CFG": CFG,
    "SEED_TRAIN": SEED_TRAIN,
    "DRIVE_OK": DRIVE_OK,
    "DRIVE_ROOT": DRIVE_ROOT,
    "ENV": ENV,
    "sh": lambda cmd, check=True, **kw: _run_command(cmd, check=check, **kw),
}


def _run_command(cmd, check=True, **kwargs):
    result = subprocess.run(cmd, text=True, capture_output=True, check=check, **kwargs)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


buf = io.StringIO()
err = None
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(train_cell, "<colab-cell-14>", "exec"), G)
except Exception as exc:
    err = f"{type(exc).__name__}: {exc}"
out = buf.getvalue()
if err:
    print("  cell raised:", err)
    print("  stdout tail:", out[-500:])
check("training cell executed without error", err is None)
check("Trainer constructed and trained in-process", "mock training done: 30 steps, batch 2" in out)
check("Drive mirror configured", "مرآة Drive مفعّلة" in out)
check("training-state patch reported", "استعادة optimizer/step/epoch مفعّلة" in out)

local_ck = sorted(RUN_DIR.glob("DiT_epoch_*_step_*.pth"))
check("three checkpoints saved locally", len(local_ck) == 3, str([p.name for p in local_ck]))
DRV_RUN = DRIVE_ROOT / "runs" / CFG["RUN_NAME"]
drv_ck = sorted(DRV_RUN.glob("DiT_epoch_*_step_*.pth")) if DRV_RUN.exists() else []
check("Drive keeps latest two checkpoints", len(drv_ck) == 2, str([p.name for p in drv_ck]))
check("final model mirrored", (DRV_RUN / "ft_model.pth").exists())
check(
    "training log written and mirrored",
    (RUN_DIR / "train_console.log").exists() and (DRV_RUN / "train_console.log").exists(),
)

# The restore cell must whitelist files and use atomic copies.
restore_cell = code_cell("١٣")
check(
    "restore cell uses atomic checkpoint copies",
    "_restore_atomic" in restore_cell and ".part" in restore_cell,
)
check(
    "restore cell filters allowed artifacts",
    "re.fullmatch" in restore_cell and "DiT_epoch_" in restore_cell,
)

hf_cell = code_cell_with("from huggingface_hub import HfApi")
check(
    "upload cell uses HfApi directly",
    "from huggingface_hub import HfApi" in hf_cell
    and "api.upload_folder(" in hf_cell
    and "subprocess.Popen" not in hf_cell,
)

print("\n" + "=" * 60)
print("IN-KERNEL TRAINING TEST:", "PASSED" if not fails else f"FAILED {fails}")
sys.exit(0 if not fails else 1)
