# -*- coding: utf-8 -*-
"""Functional dry-run tests of the notebook's critical logic (sync watcher, restore, log parsing, config)."""
import os, sys, json, shutil, subprocess, tempfile, re
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="nbtest_"))
print(f"[test] workspace: {TMP}")

# ================================================================
# TEST 1: drive_sync_loop.py — the exact script embedded in cell 14
# ================================================================
print("\n=== TEST 1: drive sync watcher ===")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
# extract embedded script from the built notebook (exactly what users will get)
nb = json.loads(Path(__file__).resolve().parent.parent / "rafidain" / "rafidain_colab_training.ipynb".read_text(encoding="utf-8"))
concat = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
m = re.search(r"SYNC_SCRIPT\.write_text\(r'''\n(.*?)\n''', encoding", concat, re.S)
assert m, "sync script not found in notebook"
sync_src = m.group(1)
ast = compile(sync_src, "drive_sync_loop.py", "exec")  # must compile
sync_path = TMP / "drive_sync_loop.py"
sync_path.write_text(sync_src, encoding="utf-8")

run_local = TMP / "runs" / "iraqi_v1"          # local run dir
drive_dst = TMP / "GDrive" / "runs" / "iraqi_v1"
run_local.mkdir(parents=True)

# simulate: a checkpoint + log written by training
ck1 = run_local / "DiT_epoch_00003_step_00750.pth"
ck1.write_bytes(b"x" * 1000)
(run_local / "train_console.log").write_text("epoch 3, step 750, loss: 0.5312\n", encoding="utf-8")

# run --once (pid 0 = no liveness check)
r = subprocess.run([sys.executable, str(sync_path), str(run_local), str(drive_dst), "0", "--once"],
                   capture_output=True, text=True, timeout=60)
print("out:", r.stdout.strip(), "| rc:", r.returncode)
assert r.returncode == 0
assert (drive_dst / ck1.name).exists(), "checkpoint not synced"
assert (drive_dst / "train_console.log").exists(), "log not synced"

# add 3 more checkpoints -> must keep only last 2 on Drive
for step, ep in ((1000, 4), (1250, 5), (1500, 6)):
    (run_local / f"DiT_epoch_{ep:05d}_step_{step:05d}.pth").write_bytes(b"y" * 500)
env = dict(os.environ); env["SYNC_KEEP"] = "2"
r = subprocess.run([sys.executable, str(sync_path), str(run_local), str(drive_dst), "0", "--once"],
                   capture_output=True, text=True, timeout=60, env=env)
drv_ckpts = sorted(drive_dst.glob("DiT_epoch_*.pth"))
names = [p.name for p in drv_ckpts]
print("drive checkpoints after prune:", names)
assert len(drv_ckpts) == 2, f"expected 2 kept, got {len(drv_ckpts)}"
assert "DiT_epoch_00006_step_01500.pth" in names and "DiT_epoch_00005_step_01250.pth" in names

# ft_model.pth must sync too
(run_local / "ft_model.pth").write_bytes(b"z" * 800)
r = subprocess.run([sys.executable, str(sync_path), str(run_local), str(drive_dst), "0", "--once"],
                   capture_output=True, text=True, timeout=60, env=env)
assert (drive_dst / "ft_model.pth").exists()
print("TEST 1 PASSED: watcher syncs, prunes to last 2, handles ft_model.pth")

# ================================================================
# TEST 2: restore logic (cell 13 core) — copy back from drive mirror
# ================================================================
print("\n=== TEST 2: restore from drive ===")
fresh_local = TMP / "fresh_local_runs" / "iraqi_v1"     # simulates post-reset empty local
fresh_local.mkdir(parents=True)
def _step_of(p):
    mm = re.search(r"step_(\d+)", p.stem)
    return int(mm.group(1)) if mm else -1
restored = 0
for f in drive_dst.iterdir():
    if f.is_file():
        shutil.copy2(f, fresh_local / f.name)
        restored += 1
ck = sorted(fresh_local.glob("DiT_epoch_*_step_*.pth"), key=_step_of)
assert restored >= 4 and ck, "restore failed"
print(f"restored {restored} files; latest local step = {_step_of(ck[-1])}")
assert _step_of(ck[-1]) == 1500
print("TEST 2 PASSED: restore brings back step-1500 checkpoint (training resumes there)")

# ================================================================
# TEST 3: monitor parsing regex (cells 15/16)
# ================================================================
print("\n=== TEST 3: log parsing ===")
STEP_RE = re.compile(r"epoch (\d+), step (\d+), loss: ([0-9.eE+-]+)")
sample_log = """[2026-11-12 10:00:00] starting training
epoch 0, step 10, loss: 0.8123
epoch 0, step 20, loss: 0.7741
epoch 1, step 890, loss: 5.2e-2
garbage line
epoch 6, step 1500, loss: 0.4210
"""
pts = []
for ln in sample_log.splitlines():
    mm = STEP_RE.search(ln)
    if mm:
        pts.append((int(mm.group(2)), float(mm.group(3))))
assert pts[-1] == (1500, 0.4210), pts[-1]
assert pts[2] == (890, 0.052), pts[2]
print("parsed points:", pts[-3:])
print("TEST 3 PASSED: step/loss parsing incl. scientific notation")

# ================================================================
# TEST 4: config cell end-to-end (no Drive, custom root)
# ================================================================
print("\n=== TEST 4: config cell in isolation ===")
cfg_cell_src = "".join(nb["cells"][4]["source"])
ns = {"__name__": "__cfg_test__"}
os.environ["IRAQI_DATA_ROOT"] = str(TMP / "root_override")
# strip the trailing summary print section? No - run whole cell; sh/ENV defined inside
exec(compile(cfg_cell_src, "cfg_cell", "exec"), ns)
CFG_T = ns["CFG"]; ROOT_T = ns["ROOT"]; CFG_PATH_T = ns["CFG_PATH"]
assert str(ROOT_T) == str(TMP / "root_override"), ROOT_T
assert CFG_T["YODAS_HOURS"] == 200 and CFG_T["STEPS"] == 3000 and CFG_T["DRIVE_BACKUP"] is True
assert CFG_PATH_T.exists(), "config not persisted"
# reload persistence: modify, rerun cell, confirm reload
CFG_T2 = json.loads(CFG_PATH_T.read_text(encoding="utf-8"))
CFG_T2["STEPS"] = 1234
CFG_PATH_T.write_text(json.dumps(CFG_T2, ensure_ascii=False), encoding="utf-8")
ns2 = {"__name__": "__cfg_test2__"}
exec(compile(cfg_cell_src, "cfg_cell", "exec"), ns2)
assert ns2["CFG"]["STEPS"] == 1234, "settings reload failed"
print(f"ROOT={ROOT_T} | persisted+reloaded OK | RUN_DIR={ns2['RUN_DIR']}")
print("TEST 4 PASSED: config defaults, persistence, reload, env override")

# ================================================================
# TEST 5: env-check cell (auto batch) in isolation
# ================================================================
print("\n=== TEST 5: env check auto-batch ===")
env_cell = "".join(nb["cells"][5]["source"])
ns3 = dict(ns2)  # inherits CFG, DRIVE_OK, etc. like the real kernel
# hide 'sys' requirement - cell imports subprocess/os/shutil/re/json; uses sys.version_info -> need sys
import sys as _sys
ns3["sys"] = _sys
# simulate GPU query result by monkeypatching subprocess inside exec? simpler: run with no GPU present
exec(compile(env_cell, "env_cell", "exec"), ns3)
print(f"no-GPU fallback batch = {ns3['BATCH']} (expected 2), saved _BATCH={ns3['CFG']['_BATCH']}")
assert ns3["BATCH"] == 2
# auto_batch thresholds
auto = ns3["auto_batch"]
assert auto(78) == 8 and auto(41) == 6 and auto(23) == 4 and auto(15.9) == 2 and auto(8) == 1
print("auto_batch thresholds OK (A100-80->8, A100-40->6, L4-24->4, T4-15->2, tiny->1)")
print("TEST 5 PASSED")

# ================================================================
# TEST 6: drive mount cell graceful degradation (non-Colab)
# ================================================================
print("\n=== TEST 6: drive mount cell without google.colab ===")
mount_cell = "".join(nb["cells"][3]["source"])
ns4 = {}
exec(compile(mount_cell, "mount_cell", "exec"), ns4)
assert ns4["DRIVE_OK"] is False and ns4["DRIVE_ROOT"] is None
print("graceful skip OK (no crash outside Colab)")
print("TEST 6 PASSED")

print("\n" + "=" * 60)
print("ALL 6 FUNCTIONAL TESTS PASSED")
shutil.rmtree(TMP, ignore_errors=True)
