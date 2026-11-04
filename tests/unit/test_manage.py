"""Unit tests for scripts/manage.py detection helpers and webui plumbing.

The manager must stay stdlib-only so it can bootstrap the environment; these
tests import it with an importlib spec (it lives outside the package).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANAGE = REPO_ROOT / "scripts" / "manage.py"


@pytest.fixture(scope="module")
def manage():
    spec = importlib.util.spec_from_file_location("rafidain_manage", MANAGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("rafidain_manage", module)
    spec.loader.exec_module(module)
    return module


class TestOsDetection:
    def test_detect_os_returns_known_value(self, manage):
        assert manage.detect_os() in ("linux", "windows", "macos", "wsl")

    def test_is_colab_false_outside_colab(self, manage, monkeypatch):
        monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
        monkeypatch.delenv("PYTHONPATH", raising=False)
        monkeypatch.setattr(manage.Path, "exists", lambda self: False)
        assert manage.is_colab() is False

    def test_is_docker_bool(self, manage):
        assert isinstance(manage.is_docker(), bool)


class TestTorchVariant:
    @pytest.mark.parametrize(
        ("cuda", "expected"),
        [
            (None, "cpu"),
            ("", "cpu"),
            ("11.8", "cu118"),
            ("11.4", "cpu"),
            ("12.1", "cpu"),  # torch 2.6 has no cu121 wheels
            ("12.4", "cu124"),
            ("12.5", "cu124"),
            ("12.6", "cu126"),
            ("12.8", "cu126"),
            ("13.0", "cu126"),
            ("garbage", "cpu"),
        ],
    )
    def test_mapping(self, manage, cuda, expected):
        assert manage.torch_variant(cuda) == expected

    def test_gpu_absent_maps_to_cpu(self, manage):
        assert manage.torch_variant(None) == "cpu"


class TestDetectGpu:
    def test_returns_none_or_dict(self, manage):
        gpu = manage.detect_gpu()
        assert gpu is None or {"name", "driver", "driver_cuda", "vram_mb"} <= set(gpu)


class TestEnvironmentReport:
    def test_report_shape(self, manage):
        report = manage.environment_report()
        assert set(report) >= {
            "os", "colab", "docker", "gpu", "torch_variant",
            "python", "uv", "ffmpeg", "repo",
        }
        assert report["repo"] == str(REPO_ROOT)

    def test_report_json_serializable(self, manage):
        import json

        json.dumps(manage.environment_report())


class TestSyncCommand:
    def test_extras_and_groups(self, manage):
        cmd = manage.sync_command(["ml", "cu124", "webui"], dev=True, frozen=True)
        assert cmd[:3] == ["uv", "sync", "--python"]
        assert "--frozen" in cmd
        assert cmd.count("--extra") == 3
        assert "ml" in cmd and "cu124" in cmd and "webui" in cmd
        assert "--group" in cmd and "dev" in cmd

    def test_no_frozen_flag_when_reresolving(self, manage):
        cmd = manage.sync_command([], dev=False, frozen=False)
        assert "--frozen" not in cmd
        assert "--group" not in cmd


class TestParser:
    def test_all_core_commands_parse(self, manage):
        parser = manage.build_parser()
        for argv in (
            ["detect"], ["status"], ["stop"],
            ["install", "--variant", "cu124"],
            ["install", "--engine", "mock"],
            ["seedvc"], ["download", "--skip-xlsr"],
            ["train", "--", "--dataset-dir", "/tmp/x"],
            ["serve", "--port", "9000"], ["web", "--tunnel"],
            ["tunnel", "start"], ["tunnel", "stop"], ["tunnel", "url"],
        ):
            assert parser.parse_args(argv).command

    def test_tunnel_token_passthrough(self, manage):
        args = manage.build_parser().parse_args(["tunnel", "start", "--token", "XYZ"])
        assert args.token == "XYZ"
