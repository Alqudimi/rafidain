#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract the scripts embedded as r'''...''' strings inside notebook cells
and syntax-check them independently."""
import ast
import re
import sys
from pathlib import Path

CELLS = Path(__file__).resolve().parent / "runpod_cells"
TMP = Path("/tmp/embedded_check")
TMP.mkdir(exist_ok=True)

# cell file -> list of (marker inside the script, output name)
targets = {
    "23_code_ab.py": [("Shared evaluation utilities", "rafidain_eval_lib.py"),
                       ("A/B evaluation: published base", "ab_eval.py")],
    "24_code_bench.py": [("Latency benchmark for the fine-tuned", "bench_stream.py")],
    "28_code_hf.py": [("import os, time", "hf_upload.py")],
}

failed = False
for cell_name, marks in targets.items():
    src = (CELLS / cell_name).read_text(encoding="utf-8")
    # find all r'''...''' blocks
    blocks = re.findall(r"r'''(.*?)'''", src, re.S)
    print(f"{cell_name}: {len(blocks)} embedded block(s)")
    for marker, out_name in marks:
        block = next((b for b in blocks if marker in b), None)
        if block is None:
            print(f"  MISSING block for {out_name} (marker: {marker[:30]})")
            failed = True
            continue
        # the embedded script starts with a shebang line; strip leading newline
        code = block.lstrip("\n")
        out = TMP / out_name
        out.write_text(code, encoding="utf-8")
        try:
            ast.parse(code)
            print(f"  OK {out_name}: {len(code.splitlines())} lines, syntax valid")
        except SyntaxError as e:
            print(f"  SYNTAX ERROR in {out_name}: line {e.lineno}: {e.msg}")
            failed = True

sys.exit(1 if failed else 0)
