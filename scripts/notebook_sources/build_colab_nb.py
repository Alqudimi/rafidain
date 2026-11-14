# -*- coding: utf-8 -*-
"""Assemble the Rafidain Google Colab training notebook and validate it thoroughly."""
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
        "title": "Rafidain - Seed-VC Iraqi Fine-tuning on Google Colab",
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
        errors.append(f"code cell #{i} (id={c['id']}): SyntaxError line {e.lineno}: {e.msg}\n   -> {src.splitlines()[e.lineno-1][:120] if e.lineno and e.lineno <= len(src.splitlines()) else '?'}")

# 3) embedded scripts inside cells must also parse
full_concat = "\n\n".join("".join(c["source"]) for c in code_cells)
for name, pat in [
    ("drive_sync_loop.py", r"SYNC_SCRIPT\.write_text\(r'''\n(.*?)\n''', encoding"),
    ("hf_upload.py",       r"HF_SCRIPT\.write_text\(r'''\n(.*?)\n''', encoding"),
]:
    m = re.search(pat, full_concat, re.S)
    if not m:
        errors.append(f"embedded {name}: pattern not found")
    else:
        try:
            ast.parse(m.group(1))
        except SyntaxError as e:
            errors.append(f"embedded {name}: SyntaxError line {e.lineno}: {e.msg}")

# 4) pipeline extraction regex must match the real repo script (both ubuntu & colab fallback URLs)
repo_sh = _HERE.parent.parent / "seed_vc_iraqi_ubuntu.sh"  # repo root
if repo_sh.exists():
    txt = repo_sh.read_text(encoding="utf-8")
    m = re.search(r"cat > \"\$ROOT/iraqi_pipeline\.py\" <<'PYEOF'\n(.*?)\nPYEOF\n", txt, re.S)
    if not m:
        errors.append("pipeline extraction regex does NOT match the real seed_vc_iraqi_ubuntu.sh")
    else:
        try:
            ast.parse(m.group(1))
            print(f"[check] pipeline extraction regex OK against repo ({len(m.group(1)):,} chars, ast OK)")
        except SyntaxError as e:
            errors.append(f"repo pipeline itself fails ast: {e}")
else:
    warnings.append("repo script not present locally; regex check skipped (was verified earlier)")

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
expected = ["١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩", "١٠", "١١", "١٢", "١٣", "١٤", "١٥", "١٦", "١٧", "١٨", "١٩", "٢٠", "٢١", "٢٢"]
got = [n for n, _ in header_nums]
if got != expected:
    errors.append(f"cell header numbering mismatch:\n  expected {expected}\n  got      {got}")
else:
    print("[check] cell numbering 1-22 consistent (Arabic-Indic)")

# 6) forbidden leftovers
for c in nb2["cells"]:
    src = "".join(c["source"])
    for bad in ("TODO", "XXX", "FIXME", "undefined", "NaN —", "placeholder"):
        if bad in src:
            warnings.append(f"cell {c['id'][:8]} contains '{bad}'")

# 7) key invariants across cells (names defined in cell 2 and used later)
must_define_in_cfg = ["def sh(", "ENV[", "CFG_PATH", "RUN_DIR", "DATA_DIR", "EXPORTS", "SEEDVC", "RAFIDAIN", "PIPE"]
cfg_src = "".join(nb2["cells"][4]["source"])  # code cell 2 = index 4 (md,md,md,code1,code2)
for token in must_define_in_cfg:
    if token not in cfg_src:
        errors.append(f"config cell missing token: {token}")

# 7b) v3.1 invariants: in-kernel deps + in-kernel training, no separate venv
full_src = full_concat
if 'sh([UV, "pip", "install", "--python", PY' not in full_src:
    errors.append("cell 5 must install deps into the session kernel via uv (--python sys.executable)")
if "import train as SEED_TRAIN" not in full_src:
    errors.append("cell 14 must train in-kernel (import train as SEED_TRAIN) — no child process")
for bad_token in ("VENV_PY", "train.pid", "sync.pid", "start_new_session"):
    if bad_token in full_src:
        errors.append(f"v3.1 must not contain '{bad_token}' (removed with venv/subprocess training)")
if "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml" not in full_src:
    errors.append("base f0 config not referenced — model swap incomplete")

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
