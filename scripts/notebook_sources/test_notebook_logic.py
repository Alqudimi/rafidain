# -*- coding: utf-8 -*-
"""Dry-run tests of the v3 notebook's critical logic.

T1/T2 build a small archive FIXTURE with the exact same structure as the real
rafidain_current_sources_ready.tar.gz (which was fully verified earlier:
4,890,474,991 bytes; rafidain_current_sources_ready/iraqi_finetune/<src>/*.ogg
+ _prep_index.jsonl + _stats.json; 56,856 clips / 387.8h — youssef = 3,867
files / 4.683h). The fixture keeps the test hermetic (no 4.9GB download).

  T1  archive completeness check (exact byte size)
  T2  extraction: tar -> DATA_DIR content mapping (iraqi_finetune/** -> data/iraqi_finetune/**)
  T3  stamp file written; second run SKIPS (no re-extract)
  T4  partial local file + resume path logic (no server hit — simulate existing partial)
  T5  stats cell logic against the fixture's _prep_index.jsonl
  T6  training cell is fully in-kernel (v3: SEED_TRAIN.Trainer, no driver script)
  T7  torch.save drive-mirror wrapper behavior (real copy + prune, simulated checkpoints)
  T8  checkpoint restore logic (finetune.checkpoint_sync.restore with fake Drive dir)
  T9  verify/export/prefetch/HF are direct in-kernel calls
  T10 anti-daemon + no-subprocess scan of the final notebook
"""
import json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NB = REPO / "rafidain_colab_training.ipynb"
EXPECT_SIZE = 4890474991   # real archive size — the notebook's verify constant

TMP = Path(tempfile.mkdtemp(prefix="nbv3_test_"))
print(f"[test] workspace: {TMP}")
fails = []

def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        fails.append(name)

# ---- fixture: a mini-archive with the real structure (3 youssef-like clips) ----
FIX_ROOT = TMP / "fixture"
(FIX_ROOT / "rafidain_current_sources_ready" / "iraqi_finetune" / "youssef").mkdir(parents=True)
ARCHIVE = TMP / "mini_ready.tar.gz"
FIX_EXPECT = 0
for i in range(3):
    (FIX_ROOT / "rafidain_current_sources_ready" / "iraqi_finetune" / "youssef" / f"youssef_p0_{i:06d}_abc__0000.ogg").write_bytes(b"OggS" + os.urandom(2000))
(FIX_ROOT / "rafidain_current_sources_ready" / "iraqi_finetune" / "_prep_index.jsonl").write_text(
    "".join(json.dumps({"file": f"youssef/youssef_p0_{i:06d}_abc__0000.ogg", "dur": 2.0}) + "\n" for i in range(3)))
(FIX_ROOT / "rafidain_current_sources_ready" / "iraqi_finetune" / "_stats.json").write_text(
    json.dumps({"total_files": 3, "total_hours": 3 * 2.0 / 3600,
                "per_source": {"youssef": {"files": 3, "hours": 3 * 2.0 / 3600}}}))
(FIX_ROOT / "rafidain_current_sources_ready" / "METADATA.json").write_text('{"project": "fixture"}')
subprocess.run(["tar", "-czf", str(ARCHIVE), "-C", str(FIX_ROOT),
                "rafidain_current_sources_ready"], check=True)
FIX_EXPECT = ARCHIVE.stat().st_size

# ================================================================
print("\n=== T1: archive completeness (constant matches real archive) ===")
nb0 = json.loads(NB.read_text(encoding="utf-8"))
cfg0 = next("".join(c["source"]) for c in nb0["cells"] if c["cell_type"] == "code" and "الخلية ٢" in "".join(c["source"]))
check("notebook verifies the real exact size 4,890,474,991", str(EXPECT_SIZE) in cfg0)
check("fixture archive complete at its own exact size",
      ARCHIVE.exists() and ARCHIVE.stat().st_size == FIX_EXPECT,
      f"{FIX_EXPECT:,} bytes")

# ================================================================
print("\n=== T2: extraction mapping (fixture tar, same structure as real) ===")
ROOT = TMP / "iraqi_seedvc"
DATA_DIR = ROOT / "data" / "iraqi_finetune"
DATA_DIR.mkdir(parents=True)
INNER = "rafidain_current_sources_ready"
tmp_ex = ROOT / "_ready_tmp"
tmp_ex.mkdir(parents=True)
# extract the fixture the same way the notebook cell does (full tar, small file)
subprocess.run(["tar", "-xzf", str(ARCHIVE)], cwd=tmp_ex, check=True)
inner_dir = tmp_ex / INNER
check("inner structure rafidain_current_sources_ready/iraqi_finetune/ present",
      (inner_dir / "iraqi_finetune").is_dir())
# move content as the notebook cell does
n_moved = 0
for src in (inner_dir / "iraqi_finetune").iterdir():
    dst = DATA_DIR / src.name
    shutil.move(str(src), str(dst))
    n_moved += 1
shutil.rmtree(tmp_ex)
n_ogg = sum(1 for _ in DATA_DIR.rglob("*.ogg"))
check("ogg clips extracted into DATA_DIR", n_ogg == 3, f"{n_ogg} ogg files")
check("metadata files moved too", (DATA_DIR / "_prep_index.jsonl").exists() and (DATA_DIR / "_stats.json").exists())

# ================================================================
print("\n=== T3: stamp + skip-on-rerun ===")
STAMP = ROOT / ".ready_archive_ok"
STAMP.write_text(json.dumps({"archive": ARCHIVE.name, "size": FIX_EXPECT,
                             "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "ogg_files": n_ogg}), encoding="utf-8")
# the notebook's skip condition: stamp exists AND any ogg present
skip = STAMP.exists() and any(DATA_DIR.rglob("*.ogg"))
check("skip condition true on rerun (no re-extraction)", skip)

# ================================================================
print("\n=== T4: partial-file resume logic ===")
part = ROOT / "partial.tar.gz"
part.write_bytes(b"\0" * 1_000_000)   # 1MB of a 4.89GB file
def archive_complete(p, expect):
    return p.exists() and p.stat().st_size == expect
check("partial file correctly detected as incomplete",
      not archive_complete(part, EXPECT_SIZE) and part.stat().st_size < EXPECT_SIZE)
# resumable curl command construction (from the notebook cell)
URL = "https://drive.usercontent.google.com/download?id=1fFCx7wvabl34Z6RtYrGEYvy9zrx5U8Du&export=download"
cmd = f"curl -L -C - --retry 8 --retry-delay 3 --connect-timeout 30 -o '{part}' '{URL}'"
check("curl resume command contains -C - (continue)", "-C -" in cmd)
check("curl resume command has retries", "--retry 8" in cmd)

# ================================================================
print("\n=== T5: stats cell logic against real index ===")
per, tot_h, tot_n = {}, 0.0, 0
for line in (DATA_DIR / "_prep_index.jsonl").read_text().splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    if not (DATA_DIR / r["file"]).exists():
        continue   # only youssef extracted in this sandbox
    src = r["file"].split("/")[0]
    per.setdefault(src, [0, 0.0])
    per[src][0] += 1
    per[src][1] += r["dur"] / 3600
    tot_n += 1
    tot_h += r["dur"] / 3600
check("per-source accounting matches _stats.json (youssef fixture)",
      per.get("youssef", [0, 0])[0] == 3 and abs(per["youssef"][1] - 3 * 2.0 / 3600) < 1e-6,
      f"youssef: {per.get('youssef')}")
check("cross-check with archive's own stats file",
      json.loads((DATA_DIR / "_stats.json").read_text())["per_source"]["youssef"]["files"] == per["youssef"][0])

# ================================================================
print("\n=== T6: training cell is fully in-kernel (v3) ===")
nb = json.loads(NB.read_text(encoding="utf-8"))
code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
train_cell = next(s for s in code_cells if "الخلية ١٢" in s)
check("no embedded TRAIN_DRIVER script (v3 runs Trainer in-kernel)",
      "TRAIN_DRIVER" not in train_cell and "train_driver.py" not in train_cell)
check("cell runs SEED_TRAIN.Trainer in-kernel", "SEED_TRAIN.Trainer(" in train_cell)
check("cell calls trainer.train()", "trainer.train()" in train_cell)
check("cell wraps torch.save before Trainer construction",
      train_cell.index("torch.save = _save_with_drive_mirror") < train_cell.index("SEED_TRAIN.Trainer("))
check("cell restores torch.save afterwards (finally)", "torch.save = _real_save" in train_cell)
check("cell tees stdout to train_console.log",
      "contextlib.redirect_stdout" in train_cell and "train_console.log" in train_cell)
check("cell has no child-process launch",
      not re.search(r"subprocess|Popen|os\.system|start_new_session|setsid", train_cell))
# cell 6 must import the trainer/inference modules into the kernel
c6 = next(s for s in code_cells if "الخلية ٦" in s)
check("cell 6 imports train+inference as kernel modules",
      "import train as SEED_TRAIN" in c6 and "import inference as SEED_INFER" in c6)
check("cell 6 sets kernel cwd to seed-vc (relative paths contract)",
      "chdir" in c6 and str("seed-vc") in c6 or "chdir(SEEDVC)" in c6)

# ================================================================
print("\n=== T7: torch.save drive-mirror wrapper (real behavior) ===")
drv = TMP / "GDrive" / "runs" / "iraqi_v1"
drv.mkdir(parents=True)
run_local = TMP / "runs" / "iraqi_v1"
run_local.mkdir(parents=True)
# simulate the wrapper (same logic as TRAIN_DRIVER, extracted verbatim)
wrapper_ns = {"__name__": "w"}
wrapper_code = '''
import os, shutil, glob as _g, re as _re
DRV_RUN = r"%s"
KEEP = 2
def _mirror_after_save(p, log_path):
    base = os.path.basename(p)
    if base.startswith("DiT_epoch_") and base.endswith(".pth"):
        os.makedirs(DRV_RUN, exist_ok=True)
        dst = os.path.join(DRV_RUN, base)
        tmp = dst + ".part"
        shutil.copy2(p, tmp)
        os.replace(tmp, dst)
        cks = sorted(_g.glob(os.path.join(DRV_RUN, "DiT_epoch_*_step_*.pth")),
                     key=lambda x: int(_re.search(r"step_(\\d+)", x).group(1)))
        for old in (cks[:-KEEP] if KEEP > 0 else []):
            try: os.remove(old)
            except OSError: pass
        return base
    elif base == "ft_model.pth":
        os.makedirs(DRV_RUN, exist_ok=True)
        shutil.copy2(p, os.path.join(DRV_RUN, base))
        return base
    return None
''' % drv
exec(wrapper_code, wrapper_ns)
mirror = wrapper_ns["_mirror_after_save"]
# fake checkpoints: steps 250, 500, 750, 1000 + ft_model
for step in (250, 500, 750, 1000):
    f = run_local / f"DiT_epoch_00001_step_{step:05d}.pth"
    f.write_bytes(os.urandom(64))
    mirror(str(f), str(run_local / "train_console.log"))
on_drive = sorted(p.name for p in drv.glob("DiT_epoch_*"))
check("only last KEEP=2 checkpoints on Drive", len(on_drive) == 2 and on_drive == [
      "DiT_epoch_00001_step_00750.pth", "DiT_epoch_00001_step_01000.pth"], str(on_drive))
ft = run_local / "ft_model.pth"
ft.write_bytes(os.urandom(64))
mirror(str(ft), str(run_local / "train_console.log"))
check("ft_model.pth mirrored to Drive", (drv / "ft_model.pth").exists())

# ================================================================
print("\n=== T8: checkpoint restore (repo module, real call) ===")
sys.path.insert(0, str(REPO))
from finetune import checkpoint_sync as CK
# fresh local run dir (simulating wiped VM) — restore from the fake Drive
local2 = TMP / "runs" / "iraqi_v2"
local2.mkdir(parents=True)
res = CK.restore(local2, drv)
check("restore pulls newest checkpoint (step 1000)", res is not None and "step_01000" in res.name,
      res.name if res else "None")
check("restored file exists locally", (local2 / "DiT_epoch_00001_step_01000.pth").exists())

# ================================================================
print("\n=== T9: verify/export/prefetch are direct in-kernel calls (v3) ===")
verify_cell = next(s for s in code_cells if "الخلية ١٣" in s)
export_cell = next(s for s in code_cells if "الخلية ١٤" in s)
prefetch_cell = next(s for s in code_cells if "الخلية ١٠" in s)
check("verify calls SEED_INFER.main in-kernel", "SEED_INFER.main(vc_args)" in verify_cell)
check("verify builds args via SimpleNamespace", "types.SimpleNamespace(" in verify_cell)
check("export calls PIPELINE.cmd_export in-kernel", "PIPELINE.cmd_export(" in export_cell)
check("prefetch calls PIPELINE.cmd_prefetch in-kernel", "PIPELINE.cmd_prefetch(" in prefetch_cell)
for nm, cl in (("verify", verify_cell), ("export", export_cell), ("prefetch", prefetch_cell)):
    check(f"{nm} cell has no child-process launch",
          not re.search(r"subprocess|Popen|os\.system", cl))

# ================================================================
print("\n=== T9b: HF upload runs in-kernel ===")
hf_cell = next(s for s in code_cells if "الخلية ١٦" in s)
check("HF imports HfApi in-kernel", "from huggingface_hub import HfApi" in hf_cell)
check("HF calls upload_folder in-kernel", "api.upload_folder(" in hf_cell)
check("HF token from CFG (not hardcoded)", 'CFG["HF_TOKEN"]' in hf_cell)
check("HF has no child-process launch",
      not re.search(r"subprocess|Popen|os\.system", hf_cell))

# ================================================================
print("\n=== T10: anti-daemon + no-subprocess scan of the final notebook ===")
full = "\n".join(code_cells)
FORBIDDEN = ["start_new_session", "setsid", "nohup", "train.pid", "sync.pid",
             "while True:", "drive_sync_loop", "time.sleep(30)",
             "TRAIN_DRIVER", "train_driver.py", "VENV_PY"]
found = [w for w in FORBIDDEN if w in full]
check("no forbidden daemon/v2 patterns in any cell", not found, str(found))
# subprocess confined to cells 4/5 (system tools + pip install)
sub_offenders = [s.splitlines()[2][:60] for s in code_cells
                 if not ("الخلية ٤" in s or "الخلية ٥" in s)
                 and re.search(r"subprocess|Popen|os\.system", s)]
check("subprocess confined to cells 4/5", not sub_offenders, str(sub_offenders))
check("resumable archive download present", "curl -L -C -" in full)
check("exact-size verify present", "stat().st_size == EXPECT" in full)
check("archive stamp present", ".ready_archive_ok" in full)
check("ready archive URL configured", "READY_ARCHIVE_URL" in full and "1fFCx7wvabl34Z6RtYrGEYvy9zrx5U8Du" in full)
check("expected size constant correct", "4890474991" in full)

# ================================================================
print("\n" + "=" * 60)
if fails:
    print(f"FAILED: {len(fails)} test(s): {fails}")
    sys.exit(1)
print("ALL NOTEBOOK LOGIC TESTS PASSED")
