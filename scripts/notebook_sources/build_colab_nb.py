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
OUT_V2 = _HERE.parent.parent / "notbooks" / "rafidain_colab_training_v2.ipynb"
OUT_V1 = _HERE.parent.parent / "notbooks" / "rafidain_colab_training.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT_V2.parent.mkdir(parents=True, exist_ok=True)

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

nb_json = json.dumps(nb, ensure_ascii=False, indent=1)
OUT.write_text(nb_json, encoding="utf-8")
OUT_V2.write_text(nb_json, encoding="utf-8")
OUT_V1.write_text(nb_json, encoding="utf-8")
print(f"[build] wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")
print(f"[build] wrote {OUT_V2} and {OUT_V1}")
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

# 3) anti-daemon policy scan — these are the patterns that get free Colab killed
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
]
for pat, why in FORBIDDEN:
    if pat in full_concat:
        # the only allowed sleep is inside retry logic of downloads — none expected here
        errors.append(f"forbidden daemon pattern present: {pat!r} ({why})")

# curl -C - resume usage is REQUIRED in the archive cell
if "curl -L -C -" not in full_concat:
    errors.append("archive cell lost its resumable download (curl -L -C -)")

# exact-size verification required
if "READY_ARCHIVE_SIZE" not in full_concat or "stat().st_size == EXPECT" not in full_concat:
    errors.append("archive cell must verify exact byte size before extraction")

# archive stamp required (no re-extraction)
if ".ready_archive_ok" not in full_concat:
    errors.append("archive stamp file (.ready_archive_ok) missing — would re-extract every run")

def _code_only(text):
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))

# training must import train as a module and call Trainer(...).train() directly without subprocess
train_cell = next(("\n".join(c["source"]) for c in code_cells if "الخلية ١٢" in "".join(c["source"])), "")
if train_cell:
    for token in ("import train as seed_train", "seed_train.Trainer(", "trainer.train()",
                  "_save_with_drive_mirror", "torch.save = _save_with_drive_mirror",
                  "ft_model.pth", "train_console.log"):
        if token not in train_cell:
            errors.append(f"training cell missing token: {token}")
    if "subprocess" in _code_only(train_cell):
        errors.append("training cell must NOT use subprocess — must call Trainer directly via import")
    if "train_driver.py" in _code_only(train_cell):
        errors.append("training cell must NOT write temporary train_driver.py script")
else:
    errors.append("training cell (12) not found")

# verify/export/prefetch cells must invoke via direct import, not subprocess or CLI
verify_cell = next(("\n".join(c["source"]) for c in code_cells if "الخلية ١٣" in "".join(c["source"])), "")
export_cell = next(("\n".join(c["source"]) for c in code_cells if "الخلية ١٤" in "".join(c["source"])), "")
prefetch_cell = next(("\n".join(c["source"]) for c in code_cells if "الخلية ١٠" in "".join(c["source"])), "")

if verify_cell:
    for token in ("import inference as inf", "inf.main(a)"):
        if token not in verify_cell:
            errors.append(f"verify cell missing token: {token}")
    if "subprocess" in _code_only(verify_cell):
        errors.append("verify cell must NOT use subprocess — must invoke inference.py directly via import")

if export_cell:
    if "cmd_export" not in export_cell:
        errors.append("export cell missing token: cmd_export")
    if "subprocess" in _code_only(export_cell):
        errors.append("export cell must NOT use subprocess — must call cmd_export directly")

if prefetch_cell:
    if "cmd_prefetch" not in prefetch_cell:
        errors.append("prefetch cell missing token: cmd_prefetch")
    if "subprocess" in _code_only(prefetch_cell):
        errors.append("prefetch cell must NOT use subprocess — must call cmd_prefetch directly")

# no CLI invocation of iraqi_pipeline.py as a script (must be imported)
if re.search(r"VENV_PY,\s*PIPE\b", full_concat) or re.search(r"PIPE\b[^)]*small", full_concat):
    errors.append("pipeline still invoked as a CLI script somewhere — must be imported")

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
must_define_in_cfg = ["def sh(", "ENV[", "CFG_PATH", "VENV_PY", "RUN_DIR", "DATA_DIR", "EXPORTS",
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
