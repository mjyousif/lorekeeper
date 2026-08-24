#!/usr/bin/env python3
"""
LoreKeeper CLI - Manage the RAG chatbot system.

Usage:
  lorekeeper start [--restart] [api|telegram|ui|all]  Start services in background
  lorekeeper stop [api|telegram|ui|all]   Stop services
  lorekeeper logs [api|telegram|ui]       Follow logs (default: api and telegram)
  lorekeeper status                       Show which services are running
  lorekeeper purge-logs                   Clear all log files before starting
  lorekeeper approve <CODE>               Approve a pairing request code
  lorekeeper help                         Show this help
"""

import logging
import os
import signal
import subprocess  # nosec
import sys
import time
from pathlib import Path
from typing import Any

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
ROOT = Path(__file__).parent.parent.parent
PID_DIR = ROOT / ".run"
LOG_DIR = ROOT / ".logs"
PID_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Service definitions
SERVICES: dict[str, dict[str, Any]] = {
    "api": {
        "cmd": [
            "uv",
            "run",
            "uvicorn",
            "src.interfaces.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        "pid": PID_DIR / "api.pid",
        "log": LOG_DIR / "api.log",
        "daemon": True,
        "depends_on": [],  # No dependencies
        "ready_url": "http://127.0.0.1:8000/",  # Health check endpoint
    },
    "telegram": {
        "cmd": ["uv", "run", "python", "-m", "src.interfaces.telegram_bot"],
        "pid": PID_DIR / "telegram.pid",
        "log": LOG_DIR / "telegram.log",
        "daemon": True,
        "depends_on": [],  # No dependencies
        "ready_url": None,  # No health check for telegram
    },
    "ui": {
        "cmd": ["uv", "run", "python", "-m", "src.interfaces.gradio_app"],
        "pid": PID_DIR / "ui.pid",
        "log": LOG_DIR / "ui.log",
        "daemon": True,
        "depends_on": ["api"],  # Wait for API to be ready
        "ready_url": None,  # No health check for UI
    },
}


def read_pid(pid_file: Path) -> int | None:
    if pid_file.exists():
        try:
            return int(pid_file.read_text().strip())
        except Exception:
            return None
    return None


def is_running(service: str) -> bool:
    pid = read_pid(SERVICES[service]["pid"])
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for_service(service: str, timeout: int = 60) -> bool:
    """Wait for a service's health check endpoint to be ready."""
    url = SERVICES[service].get("ready_url")
    if not url:
        return True  # No health check defined, assume ready if running

    logger.info(f"[{service}] Waiting for service to become ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                logger.info(f"[{service}] Service is ready!")
                return True
        except requests.RequestException:
            # Just ignore connection errors while booting
            pass
        time.sleep(1)

    logger.error(f"[{service}] Failed to become ready within {timeout}s")
    return False


def purge_logs():
    """Clear all log files in the .logs directory."""
    if LOG_DIR.exists():
        log_files = list(LOG_DIR.glob("*.log"))
        if log_files:
            logger.info(f"Purging {len(log_files)} log file(s) from {LOG_DIR}")
            for log_file in log_files:
                try:
                    log_file.unlink()
                    logger.debug(f"Removed log file: {log_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove log file {log_file}: {e}")
        else:
            logger.info("No log files to purge")
    else:
        logger.info("Log directory does not exist, skipping purge")


def start_service(service: str, restart: bool = False):
    if is_running(service):
        if restart:
            logger.info(
                f"[{service}] Already running, stopping first (restart mode)..."
            )
            stop_service(service)
        else:
            pid = read_pid(SERVICES[service]["pid"])
            logger.info(f"[{service}] Already running (PID {pid})")
            return

    # Check dependencies first
    for dep in SERVICES[service].get("depends_on", []):
        if not is_running(dep):
            logger.info(
                f"[{service}] Dependency '{dep}' is not running. Starting it first..."
            )
            start_service(dep, restart=restart)
        # Wait for dependency to be ready
        if not wait_for_service(dep):
            logger.error(f"[{service}] Dependency '{dep}' failed to start. Aborting.")
            return

    # Load environment from .env if present
    env = os.environ.copy()
    dotenv = ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k] = v

    cmd = SERVICES[service]["cmd"]
    log_path = SERVICES[service]["log"]
    pid_path = SERVICES[service]["pid"]

    logger.info(f"[{service}] Starting {' '.join(cmd)}")
    with open(log_path, "a") as log_file:
        kwargs: dict[str, Any] = {
            "cwd": ROOT,
            "env": env,
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
                | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            )
        else:
            kwargs["start_new_session"] = True

        process = subprocess.Popen(cmd, **kwargs)  # nosec
    pid_path.write_text(str(process.pid))
    time.sleep(0.5)
    if is_running(service):
        logger.info(f"[{service}] Started with PID {process.pid}")
        # Wait for this service to be ready if it has a health check
        if not wait_for_service(service):
            logger.error(
                f"[{service}] Service started but failed health check. "
                f"Check logs: {log_path}"
            )
    else:
        logger.error(f"[{service}] Failed to start (check logs: {log_path})")


def stop_service(service: str):
    pid = read_pid(SERVICES[service]["pid"])
    if pid is None:
        logger.info(f"[{service}] Not running (no PID file)")
        return
    try:
        if os.name == "nt":
            subprocess.run(  # nosec
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info(f"[{service}] Stopped process tree for PID {pid}")
        else:
            os.kill(pid, signal.SIGTERM)
            logger.info(f"[{service}] Stopped (PID {pid})")
        SERVICES[service]["pid"].unlink(missing_ok=True)
    except OSError:
        logger.warning(f"[{service}] Process {pid} not found; removing stale PID file")
        SERVICES[service]["pid"].unlink(missing_ok=True)


def tail_log(log_path: Path, follow: bool = True):
    try:
        with open(log_path, "r") as f:
            if follow:
                # Like tail -f
                f.seek(0, 2)
                while True:
                    line = f.readline()
                    if line:
                        print(line, end="")
                    else:
                        time.sleep(0.1)
            else:
                print(f.read())
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        logger.error(f"Log file not found: {log_path}")


def main(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print(__doc__)
        return 1

    command = argv[0]
    args = argv[1:]

    if command == "help":
        print(__doc__)
        return 0

    if command == "purge-logs":
        purge_logs()
        return 0

    if command == "start":
        services = args if args else ["all"]

        # Check for --restart flag in original args
        restart = "--restart" in services
        if restart:
            services.remove("--restart")

        # Expand "all" to list of services
        if "all" in services:
            services = list(SERVICES.keys())

        if restart:
            services_to_start = services.copy()
            # Stop them first so purge_logs can delete the files
            for s in services_to_start:
                if s in SERVICES:
                    stop_service(s)
            purge_logs()
            for s in services_to_start:
                if s in SERVICES:
                    start_service(s, restart=False)
        else:
            purge_logs()
            for s in services:
                if s in SERVICES:
                    start_service(s, restart=False)

        print("\n" + "=" * 45)
        print(" LoreKeeper is running! ")
        if "ui" in services and is_running("ui"):
            print(" [UI]       http://127.0.0.1:7860")
        if "api" in services and is_running("api"):
            print(" [API]      http://127.0.0.1:8000")
        if "telegram" in services and is_running("telegram"):
            print(" [Telegram] Active")
        print("=" * 45 + "\n")

    elif command == "stop":
        services = args if args else ["all"]
        if "all" in services:
            services = list(SERVICES.keys())
        for s in services:
            if s not in SERVICES:
                logger.error(f"Unknown service: {s}")
                print(__doc__)
                return 1
            stop_service(s)

    elif command == "logs":
        services = args if args else ["api", "telegram"]
        # Simple: if no services, default to both
        for s in services:
            if s not in SERVICES:
                logger.error(f"Unknown service: {s}")
                print(__doc__)
                return 1
        # If only one service, you can optionally add -f or -n flags
        # Here we always follow
        try:
            for s in services:
                print(f"--- Following {SERVICES[s]['log']} (Ctrl+C to exit) ---")
                tail_log(SERVICES[s]["log"], follow=True)
        except KeyboardInterrupt:
            pass

    elif command == "status":
        logger.info("Service Status:")
        for s in SERVICES:
            running = is_running(s)
            status = "RUNNING" if running else "STOPPED"
            pid = read_pid(SERVICES[s]["pid"])
            logger.info(f"  {s:10} {status:10} PID: {pid if pid else '-'}")

    elif command == "approve":
        if not args:
            logger.error("Usage: chatter approve <CODE>")
            return 1

        code = args[0]
        # Run the approve_pair module directly
        subprocess.run(  # nosec
            ["uv", "run", "python", "-m", "scripts.approve_pair", code], cwd=ROOT
        )

    else:
        logger.error(f"Unknown command: {command}")
        print(__doc__)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
