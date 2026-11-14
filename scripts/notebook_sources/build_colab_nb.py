# -*- coding: utf-8 -*-
"""Assemble the Rafidain Google Colab training notebook (v2) and validate it thoroughly.

v2 checks:
* no daemon patterns anywhere (Popen detached / start_new_session / PID files /
  while-True watchers / keep-alive sleep loops) — the free-tier policy killers;
* archive cell invariants (exact-size verify, stamp, Drive copy paths);
* train driver embedded code parses and imports seed-vc as a module;
* prefetch/export/verify invocation code parses;
* config cell invariants; Arabic-Indic numbering 1-18; nbformat validation.
"""
import json, ast, sys, re, uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from colab_cells_a import CELLS_A
from colab_cells_b import CELLS_B
from colab_cells_c import CELLS_C

ALL = CELLS_A + CELLS_B + CELLS_C
_HERE = Path(__file__).resolve().parent
OUT = _HERE.parent.parent / "rafidain_colab_training.ipynb"  # repo root
OUT.parent.mkdir(parents=True, exist_ok=True)

def to_source_lines(text):
    """nbformat source: list of lines each ending with \\n (except maybe last)."""
    lines = text.split("\n")
    out = []
    for i, l in enumerate(lines):
        out.append(l + ("\n" if i < len(lines) - 1 else ""))
    # drop a single trailing empty line artifact
    if out and out[-1] == "":
        out.pop()
    return out

cells = []
n_code = n_md = 0
for ctype, src in ALL:
    cell = {
        "cell_type": ctype,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": to_source_lines(src),
    }
    if ctype == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
        n_code += 1
    else:
        n_md += 1
    cells.append(cell)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
        "accelerator": "GPU",
        "title": "Rafidain - Seed-VC Iraqi Fine-tuning on Google Colab (v2, no-daemon)",
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[build] wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")
print(f"[build] {len(cells)} cells = {n_md} markdown + {n_code} code")

# ================================================================
# VALIDATION
# ================================================================
errors, warnings = [], []

# 1) JSON round-trip
nb2 = json.loads(OUT.read_text(encoding="utf-8"))
assert nb2["nbformat"] == 4

# 2) every code cell must be valid Python
code_cells = [c for c in nb2["cells"] if c["cell_type"] == "code"]
for i, c in enumerate(code_cells, 1):
    src = "".join(c["source"])
    try:
        ast.parse(src)
    except SyntaxError as e:
        bad_line = src.splitlines()[e.lineno-1][:120] if e.lineno and e.lineno <= len(src.splitlines()) else "?"
        errors.append(f"code cell #{i} (id={c['id']}): SyntaxError line {e.lineno}: {e.msg}\n   -> {bad_line}")

# 3) anti-daemon + no-subprocess policy scan
#    daemon patterns get free Colab killed; subprocess for project logic is
#    forbidden by design — everything must run inside the notebook kernel via imports.
full_concat = "\n\n".join("".join(c["source"]) for c in code_cells)
FORBIDDEN = [
    ("start_new_session", "detached process launch"),
    ("setsid",            "detached process launch"),
    ("nohup",             "background daemon"),
    ("train.pid",         "PID-file process management"),
    ("sync.pid",          "PID-file process management"),
    ("while True:",       "infinite keep-alive loop"),
    ("time.sleep(30)",    "monitoring sleep loop"),
    ("time.sleep(300)",   "watcher sleep loop"),
    ("drive_sync_loop",   "v1 background sync watcher"),
    ("deadline = time.time() + 11", "v1 11-hour wait loop"),
    ("TRAIN_DRIVER",      "v2 out-of-kernel driver script (v3 runs Trainer in-kernel)"),
    ("train_driver.py",   "v2 out-of-kernel driver script (v3 runs Trainer in-kernel)"),
    ("VENV_PY",           "v2 isolated venv (v3 installs into the session kernel)"),
    ("venv",              "isolated venv leftovers (v3 installs into the session kernel)"),
]
for pat, why in FORBIDDEN:
    if pat in full_concat:
        errors.append(f"forbidden pattern present: {pat!r} ({why})")

# 3b) subprocess usage: ONLY allowed in cells 4 (apt), 5 (pip into the kernel's own
#     python — same interpreter, package management only), and the sh() helper in
#     cell 2. Project logic (pipeline/train/inference/export/HF) must never spawn.
ALLOWED_SUBPROCESS_CELLS = ("الخلية ٤", "الخلية ٥")   # header markers
for c in code_cells:
    src = "".join(c["source"])
    if any(m in src for m in ALLOWED_SUBPROCESS_CELLS):
        continue
    for bad in ("subprocess.run", "subprocess.Popen", "subprocess.check_output",
                "subprocess.call", "os.system", "get_ipython().system("):
        if bad in src:
            line = next((l.strip()[:100] for l in src.splitlines() if bad in l), "?")
            errors.append(f"subprocess forbidden in logic cells: {bad!r} in cell {c['id'][:8]} -> {line}")

# 3c) sh() helper lives in cell 4 (system tools) — cell 2 must be subprocess-free
c4_src = "".join(code_cells[3]["source"]) if len(code_cells) > 3 else ""
if "def sh(" not in c4_src:
    errors.append("cell 4 must define sh() for system-tool commands (apt/git/tar/curl/pip)")

# curl -C - resume usage is REQUIRED in the archive cell
if "curl -L -C -" not in full_concat:
    errors.append("archive cell lost its resumable download (curl -L -)")

# exact-size verification required
if "READY_ARCHIVE_SIZE" not in full_concat or "stat().st_size == EXPECT" not in full_concat:
    errors.append("archive cell must verify exact byte size before extraction")

# archive stamp required (no re-extraction)
if ".ready_archive_ok" not in full_concat:
    errors.append("archive stamp file (.ready_archive_ok) missing — would re-extract every run")

# training must run Trainer IN-KERNEL: module imported in cell 6, called in cell 12
c6_src = "".join(code_cells[5]["source"]) if len(code_cells) > 5 else ""
for token in ("import train as SEED_TRAIN", "import inference as SEED_INFER"):
    if token not in c6_src:
        errors.append(f"cell 6 must import in-kernel module: {token}")

train_cell = next(("\n".join(c["source"]) for c in code_cells if "الخلية ١٢" in "".join(c["source"])), "")
if train_cell:
    for token in ("SEED_TRAIN.Trainer(", "trainer.train()", "contextlib.redirect_stdout",
                  "_save_with_drive_mirror", "torch.save = _save_with_drive_mirror",
                  "ft_model.pth", "train_console.log", "torch.save = _real_save"):
        if token not in train_cell:
            errors.append(f"training cell missing token: {token}")
else:
    errors.append("training cell (12) not found")

# verify/export/prefetch/HF cells must invoke in-kernel via imports
verify_cell = next(("\n".join(c["source"]) for c in code_cells if "الخلية ١٣" in "".join(c["source"])), "")
export_cell = next(("\n".join(c["source"]) for c in code_cells if "الخلية ١٤" in "".join(c["source"])), "")
prefetch_cell = next(("\n".join(c["source"]) for c in code_cells if "الخلية ١٠" in "".join(c["source"])), "")
hf_cell = next(("\n".join(c["source"]) for c in code_cells if "الخلية ١٦" in "".join(c["source"])), "")
for token in ("SEED_INFER.main(vc_args)", "types.SimpleNamespace"):
    if verify_cell and token not in verify_cell:
        errors.append(f"verify cell missing in-kernel token: {token}")
for token in ("PIPELINE.cmd_export(",):
    if export_cell and token not in export_cell:
        errors.append(f"export cell missing in-kernel token: {token}")
for token in ("PIPELINE.cmd_prefetch(",):
    if prefetch_cell and token not in prefetch_cell:
        errors.append(f"prefetch cell missing in-kernel token: {token}")
for token in ("from huggingface_hub import HfApi", "api.upload_folder("):
    if hf_cell and token not in hf_cell:
        errors.append(f"HF cell missing in-kernel token: {token}")

# 4) pipeline import path consistency: the notebook imports from the cloned repo
repo_pipeline = _HERE.parent.parent / "scripts" / "data_research" / "iraqi_pipeline_v3.py"
if repo_pipeline.exists():
    try:
        ast.parse(repo_pipeline.read_text(encoding="utf-8"))
        print(f"[check] repo pipeline parses OK ({repo_pipeline.stat().st_size:,} bytes)")
    except SyntaxError as e:
        errors.append(f"repo pipeline fails ast: {e}")
else:
    errors.append("scripts/data_research/iraqi_pipeline_v3.py missing from repo")

# 5) cross-reference consistency: cell headers vs claimed numbers
header_nums = []
for i, c in enumerate(nb2["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    m = re.search(r"#\s+الخلية\s+([٠-٩0-9]+)", src)
    if m:
        header_nums.append((m.group(1), i))
print(f"[check] code cell headers found: {[n for n, _ in header_nums]}")
expected = ["١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩", "١٠", "١١", "١٢", "١٣", "١٤", "١٥", "١٦", "١٧", "١٨"]
got = [n for n, _ in header_nums]
if got != expected:
    errors.append(f"cell header numbering mismatch:\n  expected {expected}\n  got      {got}")
else:
    print("[check] cell numbering 1-18 consistent (Arabic-Indic)")

# 6) forbidden leftovers
for c in nb2["cells"]:
    src = "".join(c["source"])
    for bad in ("TODO", "XXX", "FIXME", "undefined", "NaN —", "placeholder", "يت continue"):
        if bad in src:
            warnings.append(f"cell {c['id'][:8]} contains '{bad}'")

# 7) key invariants across cells (names defined in cell 2 and used later)
must_define_in_cfg = ["ENV[", "CFG_PATH", "RUN_DIR", "DATA_DIR", "EXPORTS",
                      "SEEDVC", "RAFIDAIN", "READY_ARCHIVE_URL", "READY_ARCHIVE_SIZE",
                      "READY_TGZ", "READY_STAMP", "drive_data_dir"]
cfg_src = "".join(nb2["cells"][4]["source"])  # code cell 2 = index 4 (md,md,md,code1,code2)
for token in must_define_in_cfg:
    if token not in cfg_src:
        errors.append(f"config cell missing token: {token}")

# 8) nbformat package validation if available
try:
    import nbformat
    nbformat.validate(nbformat.reads(OUT.read_text(encoding="utf-8"), as_version=4))
    print("[check] nbformat.validate: OK")
except ImportError:
    warnings.append("nbformat module not installed - structural JSON validation only")
except Exception as e:
    errors.append(f"nbformat.validate failed: {e}")

# ================================================================
print("\n" + "=" * 60)
if errors:
    print(f"FAILED with {len(errors)} error(s):")
    for e in errors:
        print("  ✗", e)
    sys.exit(1)
print(f"ALL CHECKS PASSED  ({len(warnings)} warning(s))")
for w in warnings:
    print("  !", w)
