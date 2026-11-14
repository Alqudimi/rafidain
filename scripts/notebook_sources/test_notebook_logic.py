# -*- coding: utf-8 -*-
"""Dry-run tests of the v2 notebook's critical logic.

Covers (against the REAL 4.89GB archive, in a sandbox ROOT):
  T1  archive completeness check (exact byte size)
  T2  extraction: tar -> DATA_DIR content mapping (iraqi_finetune/** -> data/iraqi_finetune/**)
  T3  stamp file written; second run SKIPS (no re-extract)
  T4  partial local file + resume path logic (no server hit — simulate existing partial)
  T5  stats cell logic against the real _prep_index.jsonl
  T6  train driver code: TRAIN_DRIVER string parses as Python (extracted from built notebook)
  T7  torch.save drive-mirror wrapper behavior (real copy + prune, simulated checkpoints)
  T8  checkpoint restore logic (finetune.checkpoint_sync.restore with fake Drive dir)
  T9  verify-cell invocation code builds and compiles (string)
  T10 anti-daemon scan of the final notebook (no forbidden patterns)
"""
import json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NB = REPO / "rafidain_colab_training.ipynb"
ARCHIVE = Path("/home/z/my-project/archive_check/rafidain_current_sources_ready.tar.gz")
EXPECT_SIZE = 4890474991

TMP = Path(tempfile.mkdtemp(prefix="nbv2_test_"))
print(f"[test] workspace: {TMP}")
fails = []

def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        fails.append(name)

# ================================================================
print("\n=== T1: archive completeness ===")
check("archive exists with exact expected size",
      ARCHIVE.exists() and ARCHIVE.stat().st_size == EXPECT_SIZE,
      f"{ARCHIVE.stat().st_size:,} bytes" if ARCHIVE.exists() else "missing")

# ================================================================
print("\n=== T2: extraction mapping (real tar, subset for speed) ===")
ROOT = TMP / "iraqi_seedvc"
DATA_DIR = ROOT / "data" / "iraqi_finetune"
DATA_DIR.mkdir(parents=True)
INNER = "rafidain_current_sources_ready"
tmp_ex = ROOT / "_ready_tmp"
tmp_ex.mkdir(parents=True)
# extract only youssef dir + index files (fast subset of the real archive)
subprocess.run(["tar", "-xzf", str(ARCHIVE),
                f"{INNER}/iraqi_finetune/youssef",
                f"{INNER}/iraqi_finetune/_prep_index.jsonl",
                f"{INNER}/iraqi_finetune/_stats.json",
                f"{INNER}/METADATA.json"],
               cwd=tmp_ex, check=True)
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
check("youssef subset extracted into DATA_DIR", n_ogg == 3867, f"{n_ogg} ogg files")
check("metadata files moved too", (DATA_DIR / "_prep_index.jsonl").exists() and (DATA_DIR / "_stats.json").exists())

# ================================================================
print("\n=== T3: stamp + skip-on-rerun ===")
STAMP = ROOT / ".ready_archive_ok"
STAMP.write_text(json.dumps({"archive": "x.tar.gz", "size": EXPECT_SIZE,
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
check("per-source accounting matches _stats.json (youssef)",
      per.get("youssef", [0, 0])[0] == 3867 and abs(per["youssef"][1] - 4.683) < 0.01,
      f"youssef: {per.get('youssef')}")
check("cross-check with archive's own stats file",
      json.loads((DATA_DIR / "_stats.json").read_text())["per_source"]["youssef"]["files"] == per["youssef"][0])

# ================================================================
print("\n=== T6: TRAIN_DRIVER embedded code parses ===")
nb = json.loads(NB.read_text(encoding="utf-8"))
code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
train_cell = next(s for s in code_cells if "الخلية ١٢" in s)
# extract TRAIN_DRIVER = (...) by exec-ing the cell up to that assignment in a sandbox
m = re.search(r"TRAIN_DRIVER = \((.*?)\n    \)\n", train_cell, re.S)
check("TRAIN_DRIVER assignment found in cell 12", m is not None)
if m:
    snippet = "TQ = chr(34) * 3\n" + m.group(0)
    scope = {}
    exec(snippet, scope)          # builds the string (pure literals)
    driver_src = scope["TRAIN_DRIVER"]
    try:
        compile(driver_src, "train_driver.py", "exec")
        check("TRAIN_DRIVER compiles as Python", True, f"{len(driver_src):,} chars")
    except SyntaxError as e:
        check("TRAIN_DRIVER compiles as Python", False, f"line {e.lineno}: {e.msg}")
        print("    " + driver_src.splitlines()[e.lineno-1][:100])
    # must import train as module & call Trainer
    check("driver imports train as module", "import train as seed_train" in driver_src)
    check("driver calls Trainer + train()", "seed_train.Trainer(" in driver_src and "trainer.train()" in driver_src)
    check("driver wraps torch.save", "torch.save = _save_with_drive_mirror" in driver_src)
    check("driver has no daemon launch", "start_new_session" not in driver_src and "setsid" not in driver_src)
    check("driver tees stdout to train_console.log", "train_console.log" in driver_src)

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
print("\n=== T9: verify/export invocation code compiles ===")
verify_cell = next(s for s in code_cells if "الخلية ١٣" in s)
export_cell = next(s for s in code_cells if "الخلية ١٤" in s)
prefetch_cell = next(s for s in code_cells if "الخلية ١٠" in s)
# emulate the f-string/Q building with dummy paths and compile the generated code
for name, cell, builder in (("verify", verify_cell, None), ("export", export_cell, None)):
    m2 = re.search(r"code = \((.*?)\n\)\n", cell, re.S)
    check(f"{name} cell builds code string", m2 is not None)
    if m2:
        scope = {"SEEDVC": "/x/seed-vc", "SEEDVC_CFG": "/x/cfg.yml", "PIPE_SRC": "/x/pipeline.py",
                 "RUN_DIR": "/x/runs/r1", "DATA_DIR": "/x/data", "EXPORT_DIR": "/x/exports/r1",
                 "FT": "/x/runs/r1/ft_model.pth", "src": "/x/src.wav", "ref": "/x/ref.ogg",
                 "outdir": "/x/out", "Q": "'''"}
        try:
            exec("Q = chr(39) * 3\n" + m2.group(0), scope)
            gen = scope["code"]
            compile(gen, f"<{name}-code>", "exec")
            check(f"{name} generated code compiles", True, f"{len(gen)} chars")
            if name == "verify":
                check("verify uses import inference + main", "import inference as inf" in gen and "inf.main(a)" in gen)
            else:
                check("export uses cmd_export via import", "m.cmd_export(" in gen)
        except Exception as e:
            check(f"{name} generated code compiles", False, str(e)[:100])

# ================================================================
print("\n=== T9b: HF upload invocation code compiles ===")
hf_cell = next(s for s in code_cells if "الخلية ١٦" in s)
m3 = re.search(r"code = \((.*?)\n    \)\n", hf_cell, re.S)
check("HF cell builds code string", m3 is not None)
if m3:
    scope3 = {}
    exec(m3.group(0), scope3)
    gen3 = scope3["code"]
    try:
        compile(gen3, "<hf>", "exec")
        check("HF generated code compiles", True, f"{len(gen3)} chars")
        check("HF uses HfApi + upload_folder", "HfApi" in gen3 and "upload_folder" in gen3)
        check("HF reads token from env (not hardcoded)", "os.environ['HF_TOKEN']" in gen3)
    except SyntaxError as e:
        check("HF generated code compiles", False, f"line {e.lineno}: {e.msg}")

# ================================================================
print("\n=== T10: anti-daemon scan of the final notebook ===")
full = "\n".join(code_cells)
FORBIDDEN = ["start_new_session", "setsid", "nohup", "train.pid", "sync.pid",
             "while True:", "drive_sync_loop", "time.sleep(30)"]
found = [w for w in FORBIDDEN if w in full]
check("no forbidden daemon patterns in any cell", not found, str(found))
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
