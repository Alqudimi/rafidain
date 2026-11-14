# -*- coding: utf-8 -*-
"""Execute the current Colab v3.1 setup/restore cells without Colab or a GPU.

Training and Drive-copy behavior is exercised in test_train_driver_e2e.py;
this test focuses on notebook structure, setup path wiring, no-GPU fallback,
and the restore cell's fresh-run path.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NOTEBOOK = REPO / "rafidain_colab_training.ipynb"
TMP = Path(tempfile.mkdtemp(prefix="kernel_sim_"))
print(f"[sim] workspace: {TMP}")

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
code_cells = ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"]
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def find_cell(marker: str) -> str:
    for source in code_cells:
        if marker in source:
            return source
    raise AssertionError(f"notebook cell missing marker: {marker}")


# Every code cell must remain syntactically valid after generation.
for index, source in enumerate(code_cells):
    try:
        compile(source, f"<colab-cell-{index}>", "exec")
    except SyntaxError as exc:
        check(f"code cell {index} compiles", False, f"line {exc.lineno}: {exc.msg}")
    else:
        check(f"code cell {index} compiles", True)

os.environ["IRAQI_DATA_ROOT"] = str(TMP / "iraqi_seedvc")
globals_: dict[str, object] = {"__name__": "__main__"}
sys.modules.pop("google", None)

# Cell 1 must gracefully skip Drive outside Colab.
with contextlib.redirect_stdout(io.StringIO()) as output:
    exec(find_cell("الخلية ١ — توصيل Google Drive"), globals_)
check("Drive cell runs without Colab", globals_.get("DRIVE_OK") is False)
check("Drive cell reports local-only mode", "لست على Colab" in output.getvalue())

# Cell 2 creates canonical paths and persists only the user-controlled settings.
exec(find_cell("الخلية ٢ — الإعدادات المركزية"), globals_)
required = ("ROOT", "SEEDVC", "RAFIDAIN", "DATA_DIR", "RUN_DIR", "EXPORTS", "CFG", "ENV")
check("configuration defines canonical paths", all(name in globals_ for name in required))
check("configuration has no detached venv", "VENV_PY" not in globals_)
check("configuration persisted", (TMP / "iraqi_seedvc" / "nb_config.json").is_file())
exec(find_cell("الخلية ٢ — الإعدادات المركزية"), globals_)
check("configuration cell is rerunnable", True)

# The environment cell must tolerate a machine without an NVIDIA GPU.
with contextlib.redirect_stdout(io.StringIO()) as output:
    exec(find_cell("الخلية ٣ — فحص البيئة"), globals_)
check("no-GPU fallback selects a positive batch size", int(globals_["CFG"]["_BATCH"]) >= 1)
check("no-GPU condition is reported", "لا يوجد GPU" in output.getvalue())

# The restore cell must safely report a fresh run when Drive is unavailable.
with contextlib.redirect_stdout(io.StringIO()) as output:
    exec(find_cell("الخلية ١٣ — استعادة نقاط التدريب"), globals_)
check("restore cell handles no Drive", "لا نسخة Drive" in output.getvalue())

# The current training cell uses a narrow upstream patch and no child training process.
training = find_cell("الخلية ١٤ — إطلاق التدريب داخل نواة الدفتر مباشرة")
check("training cell uses full-state patch", "ensure_full_state_resume" in training)
check(
    "training cell runs in-kernel",
    "SEED_TRAIN.Trainer" in training and "subprocess.Popen" not in training,
)
check(
    "Drive copy uses unique atomic staging",
    "tempfile.mkstemp" in training and "os.replace" in training,
)
restore = find_cell("الخلية ١٣ — استعادة نقاط التدريب")
check(
    "restore cell uses unique atomic staging",
    "tempfile.mkstemp" in restore and "os.replace" in restore,
)

print("\n" + "=" * 60)
if failures:
    print(f"FAILED: {len(failures)}: {failures}")
    sys.exit(1)
print("KERNEL SIMULATION PASSED")
