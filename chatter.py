#!/usr/bin/env python3
"""
Chatter CLI - Manage the RAG chatbot system.

Usage:
  chatter start [api|telegram|ui|all]  Start one or all services in background
  chatter stop [api|telegram|ui|all]   Stop services
  chatter logs [api|telegram]          Follow logs (default: both)
  chatter status                       Show which services are running
  Chatter help                       Show this help
"""

import os
import sys
import subprocess
import time
import signal
import json
import logging
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
ROOT = Path(__file__).parent
PID_DIR = ROOT / ".run"
LOG_DIR = ROOT / ".logs"
PID_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Service definitions
SERVICES = {
    "api": {
        "cmd": ["uv", "run", "uvicorn", "rag_wrapper.api:app", "--host", "127.0.0.1", "--port", "8000"],
        "pid": PID_DIR / "api.pid",
        "log": LOG_DIR / "api.log",
        "daemon": True,
        "depends_on": [],  # No dependencies
        "ready_url": "http://127.0.0.1:8000/",  # Health check endpoint
    },
    "telegram": {
        "cmd": ["uv", "run", "python", "telegram_bot.py"],
        "pid": PID_DIR / "telegram.pid",
        "log": LOG_DIR / "telegram.log",
        "daemon": True,
        "depends_on": ["api"],  # Wait for API to be ready
        "ready_url": None,  # No health check for telegram
    },
    "ui": {
        "cmd": ["uv", "run", "python", "gradio_app.py"],
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
        logger.debug(f"Service {service} has no health check URL, assuming ready if running")
        return True  # No health check defined, assume ready if running
    
    logger.info(f"Waiting for service {service} to become ready (timeout: {timeout}s)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                logger.info(f"Service {service} is ready (health check passed)")
                return True
        except requests.RequestException as e:
            logger.debug(f"Health check failed for {service}: {e}")
        time.sleep(1)
    
    logger.error(f"Service {service} failed to become ready within {timeout}s")
    return False


def start_service(service: str):
    if is_running(service):
        logger.info(f"[{service}] Already running (PID {read_pid(SERVICES[service]['pid'])})")
        return

    # Check dependencies first
    for dep in SERVICES[service].get("depends_on", []):
        if not is_running(dep):
            logger.info(f"[{service}] Dependency '{dep}' is not running. Starting it first...")
            start_service(dep)
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
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # daemonize (detach from parent)
        )
    pid_path.write_text(str(process.pid))
    time.sleep(0.5)
    if is_running(service):
        logger.info(f"[{service}] Started with PID {process.pid}")
        # Wait for this service to be ready if it has a health check
        if not wait_for_service(service):
            logger.error(f"[{service}] Service started but failed health check. Check logs: {log_path}")
    else:
        logger.error(f"[{service}] Failed to start (check logs: {log_path})")


def stop_service(service: str):
    pid = read_pid(SERVICES[service]["pid"])
    if pid is None:
        logger.info(f"[{service}] Not running (no PID file)")
        return
    try:
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

    if command == "start":
        services = args if args else ["all"]
        if "all" in services:
            services = list(SERVICES.keys())
        for s in services:
            if s not in SERVICES:
                logger.error(f"Unknown service: {s}")
                print(__doc__)
                return 1
            start_service(s)

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
        follow = True
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

    else:
        logger.error(f"Unknown command: {command}")
        print(__doc__)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())