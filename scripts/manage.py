#!/usr/bin/env python3
"""Rafidain unified manager — one smart CLI for the whole lifecycle.

Runs on plain Python >= 3.10 (stdlib only!) so it can bootstrap everything:
it detects the OS / GPU / Colab / Docker, installs uv, syncs the locked
environment, clones Seed-VC, downloads checkpoints, trains (with Drive
mirroring), serves the API + web UI, opens a public Cloudflare tunnel, and
stops everything cleanly.

    python scripts/manage.py detect
    python scripts/manage.py install                # uv + auto torch variant
    python scripts/manage.py seedvc download
    python scripts/manage.py train -- --dataset-dir ./data/iraqi --max-steps 2000
    python scripts/manage.py web                    # serve + /web + /live
    python scripts/manage.py tunnel start           # free trycloudflare.com URL
    python scripts/manage.py tunnel start --token EwAA...
    python scripts/manage.py status stop

Every command is also reachable through the Makefile (`make install`,
`make train ARGS="..."`, `make web`, `make tunnel`, `make stop`, ...).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / ".run"
SERVER_PID = RUN_DIR / "server.pid"
SERVER_LOG = RUN_DIR / "server.log"
TUNNEL_PID = RUN_DIR / "tunnel.pid"
TUNNEL_LOG = RUN_DIR / "tunnel.log"
TUNNEL_URL_FILE = RUN_DIR / "tunnel.url"
CLOUDFLARED = REPO / "tools" / (
    "cloudflared.exe" if os.name == "nt" else "cloudflared"
)
DEFAULT_PORT = 8000

C_CYAN, C_GREEN, C_YELLOW, C_RED, C_DIM, C_NC = (
    "\033[1;36m", "\033[1;32m", "\033[1;33m", "\033[1;31m", "\033[2m", "\033[0m"
)
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    C_CYAN = C_GREEN = C_YELLOW = C_RED = C_DIM = C_NC = ""


def say(msg: str) -> None:
    print(f"{C_CYAN}[rafidain]{C_NC} {msg}")


def ok(msg: str) -> None:
    print(f"{C_GREEN}[✓]{C_NC} {msg}")


def warn(msg: str) -> None:
    print(f"{C_YELLOW}[!]{C_NC} {msg}")


def err(msg: str) -> None:
    print(f"{C_RED}[✗]{C_NC} {msg}", file=sys.stderr)


def die(msg: str, code: int = 1) -> None:
    err(msg)
    raise SystemExit(code)


def _have_uv() -> bool:
    return shutil.which("uv") is not None


# ---------------------------------------------------------------------------
# Environment detection (pure functions — unit tested)
# ---------------------------------------------------------------------------


def detect_os() -> str:
    system = platform.system().lower()
    if system == "linux" and "microsoft" in (platform.release() or "").lower():
        return "wsl"
    return {"windows": "windows", "linux": "linux", "darwin": "macos"}.get(system, system)


def is_colab() -> bool:
    if Path("/content").is_dir() and (
        os.environ.get("COLAB_RELEASE_TAG")
        or "env/python" in (os.environ.get("PYTHONPATH") or "")
    ):
        return True
    return Path("/usr/lib/python3/dist-packages/google/colab").exists()


def is_docker() -> bool:
    return Path("/.dockerenv").exists() or (
        Path("/proc/1/cgroup").exists()
        and b"docker" in Path("/proc/1/cgroup").read_bytes()
    )


def detect_gpu() -> dict[str, Any] | None:
    """Return {'name', 'driver_cuda', 'vram_mb'} or None (no nvidia-smi)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
        if not out:
            return None
        name, driver, vram = (out.splitlines()[0].split(",") + ["", "", ""])[:3]
        cuda = ""
        ver = subprocess.run(
            [smi], capture_output=True, text=True, timeout=10, check=False
        ).stdout
        m = re.search(r"CUDA Version:\s*([0-9.]+)", ver)
        if m:
            cuda = m.group(1)
        return {
            "name": name.strip(), "driver": driver.strip(),
            "driver_cuda": cuda, "vram_mb": int(vram.strip() or 0),
        }
    except Exception:  # noqa: BLE001 — GPU probing must never crash
        return None


def torch_variant(driver_cuda: str | None) -> str:
    """Map a driver's CUDA version to the closest torch 2.6 wheel family."""
    if not driver_cuda:
        return "cpu"
    try:
        major, minor = (int(x) for x in driver_cuda.split(".")[:2])
    except ValueError:
        return "cpu"
    if major >= 13 or (major == 12 and minor >= 6):
        return "cu126"
    if major == 12 and minor >= 4:
        return "cu124"
    if major == 11 and minor >= 8:
        return "cu118"
    return "cpu"


def environment_report() -> dict[str, Any]:
    gpu = detect_gpu()
    variant = torch_variant(gpu["driver_cuda"]) if gpu else "cpu"
    py = sys.version_info
    return {
        "os": detect_os(),
        "colab": is_colab(),
        "docker": is_docker(),
        "gpu": gpu,
        "torch_variant": variant,
        "python": f"{py.major}.{py.minor}.{py.micro}",
        "uv": shutil.which("uv") or None,
        "ffmpeg": shutil.which("ffmpeg") or None,
        "git": shutil.which("git") or None,
        "repo": str(REPO),
    }


# ---------------------------------------------------------------------------
# uv bootstrap + environment sync
# ---------------------------------------------------------------------------

_UV_INSTALLERS = {
    "linux": "curl -LsSf https://astral.sh/uv/install.sh | sh",
    "macos": "curl -LsSf https://astral.sh/uv/install.sh | sh",
    "wsl": "curl -LsSf https://astral.sh/uv/install.sh | sh",
    "windows": "powershell -NoProfile -ExecutionPolicy Bypass -c "
               "'irm https://astral.sh/uv/install.ps1 | iex'",
}


def bootstrap_uv() -> str:
    """Ensure uv exists; install it when missing. Returns the uv executable."""
    if _have_uv():
        return "uv"
    osname = detect_os()
    installer = _UV_INSTALLERS.get(osname)
    if not installer:
        die(f"no automatic uv installer for {osname} — install uv manually: "
            "https://docs.astral.sh/uv/getting-started/installation/")
    say("uv not found — installing it (one time, ~30 MB)…")
    code = subprocess.call(installer, shell=True)
    if code != 0 or not _have_uv():
        # PATH may not include ~/.local/bin yet (Linux/macOS default layout)
        candidate = Path.home() / ".local" / "bin" / (
            "uv.exe" if os.name == "nt" else "uv"
        )
        if candidate.exists():
            os.environ["PATH"] = str(candidate.parent) + os.pathsep + os.environ["PATH"]
            if _have_uv():
                return "uv"
        die("uv installation failed — install it manually and re-run")
    ok("uv installed")
    return "uv"


def uv_cmd() -> str:
    return bootstrap_uv()


def sync_command(extras: list[str], *, dev: bool, frozen: bool) -> list[str]:
    """Build the `uv sync` argv honouring the locked, reproducible set."""
    cmd = ["uv", "sync", "--python", "3.11"]
    if frozen:
        cmd.append("--frozen")
    if dev:
        cmd.append("--group")
        cmd.append("dev")
    for extra in extras:
        cmd += ["--extra", extra]
    return cmd


def cmd_install(args: argparse.Namespace) -> int:
    uv_cmd()  # bootstrap uv before touching the environment
    variant = args.variant
    if variant == "auto":
        gpu = detect_gpu()
        variant = torch_variant(gpu["driver_cuda"]) if gpu else "cpu"
        if gpu:
            say(f"GPU detected: {gpu['name']} (driver CUDA {gpu['driver_cuda']}) "
                f"→ torch extra: {variant}")
        else:
            say("no NVIDIA GPU detected → torch extra: cpu (mock engine works fully)")
    extras: list[str] = []
    if args.engine == "seedvc":
        extras += ["ml", variant]
    elif args.engine == "mock":
        say("engine=mock — skipping the heavy ML stack entirely (no torch)")
    if args.webui:
        extras.append("webui")
    cmd = sync_command(extras, dev=args.dev, frozen=not args.no_frozen)
    say("environment command: " + " ".join(cmd))
    code = subprocess.call(cmd, cwd=REPO)
    if code != 0:
        die(f"uv sync failed (exit {code}) — the lock file pins every variant; "
            "check the network or run with --no-frozen to re-resolve")
    ok(f"environment ready (extras: {', '.join(extras) or 'core only'})")

    missing = [b for b in ("ffmpeg", "git") if not shutil.which(b)]
    if missing and detect_os() in ("linux", "wsl"):
        warn(f"missing system tools: {', '.join(missing)} — "
             "voices need ffmpeg: sudo apt-get install -y ffmpeg git")
    elif missing:
        warn(f"missing system tools: {', '.join(missing)} (required: ffmpeg, git)")
    if is_colab():
        say("Colab detected: training checkpoints auto-mirror to Google Drive")
    return 0


# ---------------------------------------------------------------------------
# Seed-VC + checkpoints
# ---------------------------------------------------------------------------


def cmd_seedvc(args: argparse.Namespace) -> int:
    dest = REPO / "third_party" / "seed-vc"
    if (dest / "train.py").exists() and not args.force:
        ok(f"Seed-VC already present: {dest}")
        return 0
    say("cloning Plachtaa/seed-vc (engine code + training)…")
    code = subprocess.call(
        ["git", "clone", "--depth", "1", args.repo, str(dest)], cwd=REPO
    )
    if code != 0:
        die("git clone failed — check the network")
    ok(f"Seed-VC ready at {dest}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    if not (REPO / "third_party" / "seed-vc" / "modules").exists():
        cmd_seedvc(args)
    cmd = ["uv", "run", "--python", "3.11", "python",
           "scripts/download_models.py", "--cache-dir", args.cache_dir]
    if args.skip_xlsr:
        cmd.append("--skip-xlsr")
    return subprocess.call(cmd, cwd=REPO)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    passthrough = list(args.passthrough or [])
    if not any(a.startswith("--dataset-dir") for a in passthrough):
        warn("no --dataset-dir given — run_finetune.py will refuse to start")
    if not any(a.startswith("--gdrive") for a in passthrough) and is_colab():
        passthrough += ["--gdrive", "auto"]
    cmd = ["uv", "run", "--python", "3.11", "python",
           "finetune/run_finetune.py", *passthrough]
    say("training command: " + " ".join(cmd))
    return subprocess.call(cmd, cwd=REPO)


# ---------------------------------------------------------------------------
# Serve / stop / status
# ---------------------------------------------------------------------------


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _port_from_env() -> int:
    env = os.environ.get("RAFIDAIN__PORT")
    if env and env.isdigit():
        return int(env)
    dotenv = REPO / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if line.startswith("RAFIDAIN__PORT="):
                value = line.split("=", 1)[1].strip()
                if value.isdigit():
                    return int(value)
    return DEFAULT_PORT


def cmd_serve(args: argparse.Namespace) -> int:
    RUN_DIR.mkdir(exist_ok=True)
    old = _read_pid(SERVER_PID)
    if old and _pid_alive(old):
        warn(f"server already running (pid {old}) — stop it first: manage.py stop")
        return 0
    port = args.port or _port_from_env()
    env = dict(os.environ)
    env.setdefault("RAFIDAIN__PORT", str(port))
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if args.mock:
        env["RAFIDAIN__ENGINE__IMPL"] = "mock"
    cmd = ["uv", "run", "--python", "3.11", "python", "-m", "rafidain.main"]
    say(f"starting server on :{port} (log: {SERVER_LOG})")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with open(SERVER_LOG, "ab") as log_fh:
        proc = subprocess.Popen(
            cmd, cwd=REPO, env=env, stdout=log_fh,
            stderr=subprocess.STDOUT, creationflags=flags,
        )
    SERVER_PID.write_text(str(proc.pid))
    deadline = time.time() + args.wait
    while time.time() < deadline:
        if _http_ok(f"http://127.0.0.1:{port}/healthz"):
            ok(f"server is up: http://127.0.0.1:{port}")
            say(f"  control panel : http://127.0.0.1:{port}/web")
            say(f"  live streaming: http://127.0.0.1:{port}/live")
            say(f"  API docs      : http://127.0.0.1:{port}/docs")
            return 0
        if proc.poll() is not None:
            err("server exited early — last log lines:")
            tail = SERVER_LOG.read_bytes()[-2000:].decode("utf-8", "replace")
            print(tail, file=sys.stderr)
            return proc.returncode or 1
        time.sleep(1)
    warn(f"health check timed out after {args.wait}s — inspect {SERVER_LOG}")
    return 1


def cmd_web(args: argparse.Namespace) -> int:
    args.mock = getattr(args, "mock", False)
    if not getattr(args, "mock", False):
        # the web panel needs gradio — make sure the extra is synced in
        say("ensuring the webui extra is installed (gradio)…")
        code = subprocess.call(
            ["uv", "sync", "--python", "3.11", "--frozen", "--extra", "webui"],
            cwd=REPO,
        )
        if code != 0:
            warn("could not sync webui extra — the panel at /web needs gradio")
    rc = cmd_serve(args)
    if rc == 0 and args.tunnel:
        return tunnel_start(port=args.port or _port_from_env(), token=args.token)
    return rc


def _http_ok(url: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def _http_json(url: str) -> dict | None:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def cmd_stop(args: argparse.Namespace) -> int:
    RUN_DIR.mkdir(exist_ok=True)
    stopped = []
    for name, path in (("server", SERVER_PID), ("tunnel", TUNNEL_PID)):
        pid = _read_pid(path)
        if pid and _pid_alive(pid):
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"],
                                   capture_output=True)
                else:
                    os.kill(pid, 15)
                stopped.append(f"{name} (pid {pid})")
            except OSError as exc:
                warn(f"could not stop {name} pid {pid}: {exc}")
        path.unlink(missing_ok=True)
    # best-effort sweep of strays started outside manage.py
    if args.sweep and os.name != "nt":
        subprocess.call(["pkill", "-f", "rafidain.main"], stderr=subprocess.DEVNULL)
        subprocess.call(["pkill", "-f", "cloudflared tunnel"], stderr=subprocess.DEVNULL)
        stopped.append("(swept any strays)")
    if stopped:
        ok("stopped: " + ", ".join(stopped))
    else:
        say("nothing to stop (no pid files)")
    TUNNEL_URL_FILE.unlink(missing_ok=True)
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    RUN_DIR.mkdir(exist_ok=True)
    print(f"{C_CYAN}── رافدَين — حالة النظام ──{C_NC}")
    env = environment_report()
    say(f"النظام: {env['os']}" + (" · Colab" if env["colab"] else "")
        + (" · Docker" if env["docker"] else ""))
    if env["gpu"]:
        say(f"الكرت: {env['gpu']['name']} · CUDA {env['gpu']['driver_cuda']} "
            f"· {env['gpu']['vram_mb']}MB → torch {env['torch_variant']}")
    else:
        say("الكرت: لا يوجد NVIDIA GPU")
    say(f"ffmpeg: {env['ffmpeg'] or 'غير مثبت!'} · uv: {env['uv'] or 'غير مثبت'}")

    port = _port_from_env()
    pid = _read_pid(SERVER_PID)
    if pid and _pid_alive(pid):
        health = _http_json(f"http://127.0.0.1:{port}/healthz")
        ready = _http_json(f"http://127.0.0.1:{port}/readyz")
        say(f"الخادم: يعمل (pid {pid}) على :{port} · "
            f"{health.get('status') if health else 'لا استجابة'} · "
            f"جاهز: {ready.get('ready') if ready else '?'}")
        say(f"  لوحة التحكم: http://127.0.0.1:{port}/web · "
            f"البث الحي: http://127.0.0.1:{port}/live")
    else:
        say("الخادم: متوقف (شغّله: make web)")

    tunnel_url = tunnel_url_value()
    if tunnel_url:
        say(f"النفق العام: {tunnel_url}")
    elif _read_pid(TUNNEL_PID):
        say("النفق: يعمل لكن الرابط غير معروف بعد — جرب manage.py tunnel url")
    else:
        say("النفق: متوقف (افتحه: make tunnel)")
    return 0


# ---------------------------------------------------------------------------
# Cloudflare tunnel
# ---------------------------------------------------------------------------

_CF_PLATFORMS = {
    ("linux", "x86_64"): "cloudflared-linux-amd64",
    ("linux", "aarch64"): "cloudflared-linux-arm64",
    ("darwin", "x86_64"): "cloudflared-darwin-amd64.tgz",
    ("darwin", "arm64"): "cloudflared-darwin-arm64.tgz",
    ("windows", "AMD64"): "cloudflared-windows-amd64.exe",
    ("windows", "x86_64"): "cloudflared-windows-amd64.exe",
}
_TUNNEL_URL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


def _ensure_cloudflared() -> Path:
    if CLOUDFLARED.exists():
        return CLOUDFLARED
    key = (detect_os(), platform.machine())
    asset = _CF_PLATFORMS.get(key)
    if not asset:
        die(f"no cloudflared build for {key} — install it manually: "
            "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")
    CLOUDFLARED.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/{asset}"
    say(f"downloading cloudflared ({asset})…")
    import urllib.request

    tmp = CLOUDFLARED.with_suffix(".download")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    if asset.endswith(".tgz"):
        import tarfile

        with tarfile.open(tmp) as tar:
            member = [m for m in tar.getmembers() if m.name.endswith("cloudflared")][0]
            with open(CLOUDFLARED, "wb") as out:
                shutil.copyfileobj(tar.extractfile(member), out)  # noqa: S202
        tmp.unlink(missing_ok=True)
    else:
        tmp.rename(CLOUDFLARED)
    CLOUDFLARED.chmod(0o755)
    ok(f"cloudflared ready: {CLOUDFLARED}")
    return CLOUDFLARED


def tunnel_url_value() -> str | None:
    try:
        url = TUNNEL_URL_FILE.read_text().strip()
        return url or None
    except OSError:
        return None


def tunnel_start(port: int, token: str | None = None) -> int:
    RUN_DIR.mkdir(exist_ok=True)
    old = _read_pid(TUNNEL_PID)
    if old and _pid_alive(old):
        warn(f"tunnel already running (pid {old}) — stop it: manage.py tunnel stop")
        return 0
    if not _http_ok(f"http://127.0.0.1:{port}/healthz"):
        die(f"no server on :{port} — start it first (make web / make serve)")
    binary = _ensure_cloudflared()
    if token:
        cmd = [str(binary), "tunnel", "run", "--token", token]
        say("starting a NAMED Cloudflare tunnel (your own domain)…")
    else:
        cmd = [str(binary), "tunnel", "--url", f"http://127.0.0.1:{port}",
               "--no-autoupdate"]
        say("starting a FREE quick tunnel → https://<random>.trycloudflare.com …")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with open(TUNNEL_LOG, "ab") as log_fh:
        proc = subprocess.Popen(
            cmd, cwd=REPO, stdout=log_fh, stderr=subprocess.STDOUT,
            creationflags=flags,
        )
    TUNNEL_PID.write_text(str(proc.pid))
    deadline = time.time() + 90
    while time.time() < deadline:
        log_text = TUNNEL_LOG.read_text(encoding="utf-8", errors="replace") \
            if TUNNEL_LOG.exists() else ""
        match = _TUNNEL_URL_RE.search(log_text)
        if match:
            url = match.group(0)
            TUNNEL_URL_FILE.write_text(url)
            ok(f"الرابط العام (مؤقت ومجاني): {url}")
            say(f"  لوحة التحكم: {url}/web")
            say(f"  البث الحي   : {url}/live?token=<مفتاحك>"
                if os.environ.get("RAFIDAIN__API_KEYS") else f"  البث الحي   : {url}/live")
            say("  احفظ الرابط — يُولَّد عنوان جديد عند كل تشغيل")
            return 0
        if proc.poll() is not None:
            err("cloudflared exited early — log tail:")
            print(log_text[-1500:], file=sys.stderr)
            return proc.returncode or 1
        time.sleep(1)
    warn("tunnel started but no URL parsed yet — check: .run/tunnel.log")
    return 1


def cmd_tunnel(args: argparse.Namespace) -> int:
    action = args.action
    if action == "start":
        token = args.token or os.environ.get("CLOUDFLARED_TOKEN") or None
        return tunnel_start(port=args.port or _port_from_env(), token=token)
    if action == "stop":
        pid = _read_pid(TUNNEL_PID)
        if pid and _pid_alive(pid):
            with contextlib.suppress(OSError):
                os.kill(pid, 15)
            ok(f"tunnel stopped (pid {pid})")
        else:
            say("tunnel not running")
        TUNNEL_PID.unlink(missing_ok=True)
        TUNNEL_URL_FILE.unlink(missing_ok=True)
        return 0
    if action == "url":
        url = tunnel_url_value()
        print(url if url else "(no active tunnel URL — run: manage.py tunnel start)")
        return 0
    if action == "log":
        if TUNNEL_LOG.exists():
            print(TUNNEL_LOG.read_text(encoding="utf-8", errors="replace")[-4000:])
        else:
            say("no tunnel log yet")
        return 0
    if action == "status":
        pid = _read_pid(TUNNEL_PID)
        if pid and _pid_alive(pid):
            say(f"tunnel running (pid {pid}) → {tunnel_url_value() or 'URL pending'}")
        else:
            say("tunnel stopped")
        return 0
    die(f"unknown tunnel action: {action}")
    return 1


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="manage.py",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("detect", help="print the smart environment report") \
        .set_defaults(func=lambda a: (print(json.dumps(environment_report(),
                                               ensure_ascii=False, indent=2)), 0)[1])
    sub.add_parser("env-info", help="same as detect (friendly alias)") \
        .set_defaults(func=lambda a: (print(json.dumps(environment_report(),
                                               ensure_ascii=False, indent=2)), 0)[1])

    p = sub.add_parser("install", help="bootstrap uv + sync the locked environment")
    p.add_argument("--engine", choices=("seedvc", "mock", "auto"), default="auto",
                   help="which stack: seedvc needs the ML deps (default: auto)")
    p.add_argument("--variant", choices=("auto", "cpu", "cu118", "cu124", "cu126"),
                   default="auto", help="torch wheel family (default: auto-detect GPU)")
    p.add_argument("--no-webui", dest="webui", action="store_false", default=True,
                   help="skip the gradio extra")
    p.add_argument("--dev", action="store_true", help="include the dev tooling group")
    p.add_argument("--no-frozen", action="store_true",
                   help="allow re-resolution instead of the committed lock")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("seedvc", help="clone the Seed-VC repository (third_party/)")
    p.add_argument("--repo", default="https://github.com/Plachtaa/seed-vc")
    p.add_argument("--force", action="store_true", help="re-clone even if present")
    p.set_defaults(func=cmd_seedvc)

    p = sub.add_parser("download", help="pre-download model checkpoints")
    p.add_argument("--cache-dir", default="./checkpoints")
    p.add_argument("--skip-xlsr", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("train", help="run fine-tuning (args after -- pass through)")
    p.add_argument("passthrough", nargs=argparse.REMAINDER,
                   help="e.g. -- --dataset-dir ./data --run-name iraqi_v1")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("serve", help="start the API server in the background")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--mock", action="store_true", help="force the mock engine")
    p.add_argument("--wait", type=int, default=120, help="health wait seconds")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("web", help="serve with the web UI (+ optional tunnel)")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--tunnel", action="store_true", help="also open the public tunnel")
    p.add_argument("--token", default=None, help="named-tunnel token (optional)")
    p.add_argument("--wait", type=int, default=120)
    p.set_defaults(func=cmd_web)

    p = sub.add_parser("stop", help="stop the server and the tunnel")
    p.add_argument("--sweep", action="store_true", help="also pkill strays (unix)")
    p.set_defaults(func=cmd_stop)

    sub.add_parser("status", help="health / engine / tunnel report") \
        .set_defaults(func=cmd_status)

    p = sub.add_parser("tunnel", help="manage the public Cloudflare tunnel")
    p.add_argument("action", choices=("start", "stop", "status", "url", "log"),
                   nargs="?", default="start")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--token", default=None,
                   help="named tunnel token (falls back to $CLOUDFLARED_TOKEN)")
    p.set_defaults(func=cmd_tunnel)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # tolerate `manage.py train --dataset-dir x` (no leading --)
    if args.command == "train" and args.passthrough and args.passthrough[0] != "--":
        pass  # REMAINDER already captured them
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
