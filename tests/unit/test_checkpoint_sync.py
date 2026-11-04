"""Unit tests for finetune/checkpoint_sync.py (no torch, no Drive needed)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from finetune.checkpoint_sync import (
    CheckpointSync,
    _copy_atomic,
    _step_of,
    default_remote_root,
    latest_checkpoint,
    push_once,
    restore,
    run_dir_for,
)


def _make_ckpt(directory: Path, step: int, *, size: int = 64, epoch: int = 1) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"DiT_epoch_{epoch:05d}_step_{step:05d}.pth"
    path.write_bytes(b"x" * size)
    return path


class TestLatestCheckpoint:
    def test_empty_and_missing(self, tmp_path: Path):
        assert latest_checkpoint(tmp_path) is None
        assert latest_checkpoint(tmp_path / "nope") is None

    def test_newest_step_wins(self, tmp_path: Path):
        _make_ckpt(tmp_path, 100)
        newest = _make_ckpt(tmp_path, 900)
        _make_ckpt(tmp_path, 300)
        assert latest_checkpoint(tmp_path) == newest

    def test_ignores_ft_model(self, tmp_path: Path):
        (tmp_path / "ft_model.pth").write_bytes(b"y" * 8)
        assert latest_checkpoint(tmp_path) is None

    def test_step_parsing(self, tmp_path: Path):
        p = _make_ckpt(tmp_path, 4242)
        assert _step_of(p) == 4242


class TestRestore:
    def test_no_remote_returns_local(self, tmp_path: Path):
        local = tmp_path / "local"
        ckpt = _make_ckpt(local, 500)
        assert restore(local, None) == ckpt
        assert restore(local, tmp_path / "missing-remote") == ckpt

    def test_pulls_newer_remote(self, tmp_path: Path):
        local, remote = tmp_path / "local", tmp_path / "remote"
        _make_ckpt(local, 100)
        remote_best = _make_ckpt(remote, 800)
        result = restore(local, remote)
        assert result == local / remote_best.name
        assert (local / remote_best.name).exists()

    def test_restore_also_bring_aux_files(self, tmp_path: Path):
        local, remote = tmp_path / "local", tmp_path / "remote"
        _make_ckpt(remote, 50)
        (remote / "config_patched.yml").write_text("log_dir: ./runs/")
        restore(local, remote)
        assert (local / "config_patched.yml").read_text() == "log_dir: ./runs/"

    def test_skips_when_local_newer(self, tmp_path: Path):
        local, remote = tmp_path / "local", tmp_path / "remote"
        local_best = _make_ckpt(local, 950)
        _make_ckpt(remote, 100)
        assert restore(local, remote) == local_best
        assert not (local / "DiT_epoch_00001_step_00100.pth").exists()


class TestPushOnce:
    def test_mirrors_and_prunes(self, tmp_path: Path):
        local, remote = tmp_path / "local", tmp_path / "remote"
        for step in (100, 200, 300, 400, 500):
            _make_ckpt(local, step)
        copied = push_once(local, remote, keep_last=2)
        assert len(copied) == 5
        names = sorted(p.name for p in remote.glob("DiT_epoch_*.pth"))
        # pruned to the newest 2 remotely
        assert len(names) == 2
        assert "DiT_epoch_00001_step_00500.pth" in names
        # local untouched (upstream manages its own window)
        assert len(list(local.glob("DiT_epoch_*.pth"))) == 5

    def test_second_push_is_noop(self, tmp_path: Path):
        local, remote = tmp_path / "local", tmp_path / "remote"
        _make_ckpt(local, 10)
        push_once(local, remote, keep_last=3)
        assert push_once(local, remote, keep_last=3) == []

    def test_final_included_on_demand(self, tmp_path: Path):
        local, remote = tmp_path / "local", tmp_path / "remote"
        _make_ckpt(local, 10)
        (local / "ft_model.pth").write_bytes(b"f" * 32)
        push_once(local, remote, include_final=True)
        assert (remote / "ft_model.pth").exists()

    def test_atomic_copy_leaves_no_part_files(self, tmp_path: Path):
        src = _make_ckpt(tmp_path / "src", 5)
        dst_dir = tmp_path / "dst"
        _copy_atomic(src, dst_dir / "five.pth")
        assert (dst_dir / "five.pth").exists()
        assert list(dst_dir.glob("*.part")) == []


class TestCheckpointSyncThread:
    def test_start_stop_fast_interval(self, tmp_path: Path, monkeypatch):
        local, remote = tmp_path / "local", tmp_path / "remote"
        _make_ckpt(local, 700)
        logs: list[str] = []
        sync = CheckpointSync(
            local, remote, interval_s=5.0, keep_last=3, log=logs.append
        )
        # shrink the floor so the thread fires quickly in tests
        monkeypatch.setattr(sync, "interval_s", 0.05)
        assert sync.enabled
        assert sync.restore_first() is not None
        sync.start()
        time.sleep(0.2)
        _make_ckpt(local, 900)
        time.sleep(0.25)
        sync.stop(final=True)
        assert (remote / "DiT_epoch_00001_step_00900.pth").exists()
        assert (remote / "ft_model.pth").exists() is False  # no final artefact
        assert any("background mirror started" in m for m in logs)

    def test_disabled_without_remote(self, tmp_path: Path):
        sync = CheckpointSync(tmp_path / "local", None)
        assert not sync.enabled
        sync.start()  # no-op
        sync.stop(final=True)  # no-op, no crash


class TestDriveDetection:
    def test_env_override(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("RAFIDAIN_GDRIVE_DIR", str(tmp_path))
        from finetune.checkpoint_sync import detect_google_drive

        assert detect_google_drive() == tmp_path

    def test_env_override_missing_dir(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("RAFIDAIN_GDRIVE_DIR", str(tmp_path / "void"))
        from finetune.checkpoint_sync import detect_google_drive

        assert detect_google_drive(auto_mount=False) is None

    def test_default_remote_root(self, tmp_path: Path):
        assert default_remote_root(None) is None
        assert default_remote_root(tmp_path).name == "RafidainRuns"

    def test_run_dir_layout(self, tmp_path: Path):
        assert run_dir_for(tmp_path, "iraqi_v1") == tmp_path / "runs" / "iraqi_v1"


class TestAtomicSafety:
    def test_partial_part_file_is_overwritten(self, tmp_path: Path):
        local, remote = tmp_path / "local", tmp_path / "remote"
        ckpt = _make_ckpt(local, 10)
        remote.mkdir()
        (remote / (ckpt.name + ".part")).write_bytes(b"garbage")
        push_once(local, remote, keep_last=3)
        assert (remote / ckpt.name).read_bytes() == b"x" * 64
        assert not (remote / (ckpt.name + ".part")).exists()


@pytest.mark.parametrize("mtime_drift", [0, 1])
def test_same_file_detection_by_size_and_mtime(tmp_path: Path, mtime_drift: int):
    local, remote = tmp_path / "l", tmp_path / "r"
    ckpt = _make_ckpt(local, 10)
    remote.mkdir()
    _copy_atomic(ckpt, remote / ckpt.name)
    if mtime_drift:
        past = time.time() - 7200
        os.utime(remote / ckpt.name, (past, past))
    copied = push_once(local, remote, keep_last=3)
    assert len(copied) == (1 if mtime_drift else 0)
