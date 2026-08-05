"""Launch the EvoMOBO Streamlit app."""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = Path(__file__).resolve().with_name("streamlit_app.py")
DEFAULT_PORT = 6789


def is_headless_server() -> bool:
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return True
    if sys.platform.startswith("linux"):
        return not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
    return False


def resolve_mode(requested_mode: str) -> str:
    if requested_mode == "auto":
        return "remote" if is_headless_server() else "local"
    return requested_mode


def detect_server_host() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            address = sock.getsockname()[0]
            if address and not address.startswith("127."):
                return address
    except OSError:
        pass

    try:
        for address in socket.gethostbyname_ex(socket.gethostname())[2]:
            if address and not address.startswith("127."):
                return address
    except OSError:
        pass
    return "<server-ip-or-hostname>"


def firewall_command(port: int) -> str:
    if shutil.which("firewall-cmd"):
        return (
            f"sudo firewall-cmd --permanent --add-port={port}/tcp "
            "&& sudo firewall-cmd --reload"
        )
    if shutil.which("ufw"):
        return f"sudo ufw allow {port}/tcp"
    return f"Allow TCP port {port} in the server firewall or security group."


def open_local_browser(url: str) -> bool:
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        print(
            "Browser launch skipped: no graphical desktop session was "
            "detected on the server."
        )
        return False

    try:
        opened = webbrowser.open(url, new=2)
    except webbrowser.Error:
        opened = False

    if opened:
        print(f"Server browser opened: {url}")
    else:
        print(
            "Browser launch failed: no usable default browser was detected "
            "on the server."
        )
    return opened


def run_streamlit(
    command: list[str],
    environment: dict[str, str],
    address: str,
    port: int,
    local_url: str | None = None,
) -> int:
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    hidden_lines = {
        "You can now view your Streamlit app in your browser.",
        f"URL: http://{address}:{port}",
    }
    browser_attempted = False
    for line in process.stdout:
        stripped = line.strip()
        if (
            local_url
            and not browser_attempted
            and stripped == f"URL: http://{address}:{port}"
        ):
            browser_attempted = True
            open_local_browser(local_url)
        if stripped not in hidden_lines:
            print(line, end="", flush=True)
    return process.wait()


def replace_with_streamlit(
    command: list[str],
    environment: dict[str, str],
) -> None:
    """Run the remote server as the foreground process on POSIX systems."""
    os.chdir(PROJECT_ROOT)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, command, environment)


def build_streamlit_command(
    mode: str,
    address: str,
    port: int,
    public_host: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        f"--server.address={address}",
        f"--server.port={port}",
        "--server.headless=true",
        "--server.runOnSave=false",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]
    if mode == "remote":
        command.extend(
            [
                f"--browser.serverAddress={public_host}",
                f"--browser.serverPort={port}",
                "--server.enableCORS=true",
                "--server.enableXsrfProtection=true",
                "--server.enableWebsocketCompression=false",
            ]
        )
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the EvoMOBO Streamlit interface."
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "local", "remote"),
        default="auto",
        help=(
            "Select local or remote access. Auto uses remote mode on a "
            "headless or SSH server and local mode on a desktop."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="TCP port used by the Streamlit server.",
    )
    parser.add_argument(
        "--public-host",
        default="",
        help="Optional public IP address or hostname shown in remote instructions.",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    return args


def main() -> int:
    args = parse_args()
    mode = resolve_mode(args.mode)
    address = "127.0.0.1" if mode == "local" else "0.0.0.0"
    public_host = args.public_host.strip()
    if mode == "remote" and not public_host:
        public_host = detect_server_host()
    environment = os.environ.copy()
    environment.update(
        {
            "EVOMOBO_STREAMLIT_MODE": mode,
            "EVOMOBO_STREAMLIT_ADDRESS": address,
            "EVOMOBO_STREAMLIT_PORT": str(args.port),
            "EVOMOBO_STREAMLIT_PUBLIC_HOST": public_host,
            "PYTHONFAULTHANDLER": "1",
        }
    )

    if args.mode == "auto":
        print(f"EvoMOBO automatically selected {mode} mode.")

    if mode == "remote":
        print("=" * 72)
        print("EvoMOBO Streamlit remote mode")
        print(f"Remote client URL: http://{public_host}:{args.port}")
        print(f"Open the port: {firewall_command(args.port)}")
        print("Risk: expose this port only to trusted client IP addresses.")
        print("=" * 72)
    else:
        print("=" * 72)
        print("EvoMOBO Streamlit local mode")
        print(f"Server-local URL: http://127.0.0.1:{args.port}")
        print("Browser: the launcher will try to open the server's default browser.")
        print("=" * 72)

    command = build_streamlit_command(
        mode=mode,
        address=address,
        port=args.port,
        public_host=public_host,
    )
    sys.stdout.flush()
    if mode == "remote" and os.name == "posix":
        replace_with_streamlit(command, environment)
        raise AssertionError("os.execve returned unexpectedly")

    local_url = (
        f"http://127.0.0.1:{args.port}" if mode == "local" else None
    )
    return run_streamlit(
        command,
        environment,
        address,
        args.port,
        local_url=local_url,
    )


if __name__ == "__main__":
    raise SystemExit(main())
