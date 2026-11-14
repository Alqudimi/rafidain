"""Guarded Seed-VC compatibility patches required for reliable training.

Upstream currently discovers a rolling training checkpoint but loads it as
weights-only, resetting optimizer, scheduler, epoch and step counters. The
patch below restores full state only for a rolling checkpoint inside the
current run directory; published or user-supplied initial weights remain a
weights-only warm start. It also makes an already-completed resume finalize
without executing one extra optimizer step.
"""

from __future__ import annotations

from pathlib import Path

_MARKER = "# RAFIDAIN_FULL_STATE_RESUME"
_OLD_LOAD = """        if os.path.exists(latest_checkpoint):
            self.model, self.optimizer, self.epoch, self.iters = load_checkpoint(
                self.model, self.optimizer, latest_checkpoint,
                load_only_params=True,
                ignore_modules=[],
                is_distributed=False
            )"""
_NEW_LOAD = """        if os.path.exists(latest_checkpoint):
            # RAFIDAIN_FULL_STATE_RESUME: rolling run checkpoints carry optimizer state.
            is_training_resume = (
                pretrained_ckpt_path is None
                and os.path.basename(latest_checkpoint).startswith("DiT_epoch_")
                and os.path.commonpath([
                    os.path.abspath(latest_checkpoint), os.path.abspath(self.log_dir)
                ]) == os.path.abspath(self.log_dir)
            )
            self.model, self.optimizer, self.epoch, self.iters = load_checkpoint(
                self.model, self.optimizer, latest_checkpoint,
                load_only_params=not is_training_resume,
                ignore_modules=[],
                is_distributed=False
            )"""
_OLD_TRAIN = """    def train(self):
        self.ema_loss = 0
        self.loss_smoothing_rate = 0.99
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            self.train_one_epoch()
            if self.iters >= self.max_steps:
                break
"""
_NEW_TRAIN = """    def train(self):
        self.ema_loss = 0
        self.loss_smoothing_rate = 0.99
        # RAFIDAIN_FULL_STATE_RESUME: do not run an extra batch when already complete.
        if self.iters < self.max_steps:
            for epoch in range(self.epoch, self.n_epochs):
                self.epoch = epoch
                self.train_one_epoch()
                if self.iters >= self.max_steps:
                    break
"""


def ensure_full_state_resume(seed_vc_root: Path) -> bool:
    """Apply the known patch once; refuse unknown upstream code layouts.

    Returns True when patched and False when the markers already exist. The
    operation is atomic, repeatable, and deliberately fails closed if upstream
    changes the trainer source so the patch cannot be safely reviewed.
    """
    train_py = Path(seed_vc_root) / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(f"Seed-VC train.py not found: {train_py}")
    source = train_py.read_text(encoding="utf-8")
    if _MARKER in source:
        if "if self.iters < self.max_steps:" not in source:
            raise RuntimeError(
                "partial Rafidain upstream patch detected; inspect train.py before continuing"
            )
        return False
    if source.count(_OLD_LOAD) != 1 or source.count(_OLD_TRAIN) != 1:
        raise RuntimeError(
            "Unsupported Seed-VC train.py layout; refusing to patch blindly. "
            "Pin a compatible upstream commit or review the checkpoint loader."
        )
    updated = source.replace(_OLD_LOAD, _NEW_LOAD, 1).replace(_OLD_TRAIN, _NEW_TRAIN, 1)
    temp = train_py.with_name(train_py.name + ".rafidain-tmp")
    temp.write_text(updated, encoding="utf-8")
    temp.replace(train_py)
    return True
