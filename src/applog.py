"""
Async, low-overhead logging for the engagement + physio pipeline.

One shared setup function used by every entry-point script
(`live_multiperson_binary_v2.py`, `multiemotibit_UDP_SD_RFv2.py`,
`redis_subscriber_all_devices.py`, ...). Each script calls
`setup_logging("<name>")` once at startup; afterwards any module
can do `logging.getLogger(__name__).info(...)` and it Just Works.

Design:
  - Producer threads (camera loop, UDP receiver, Redis subscriber)
    only push log records onto an in-process queue.Queue via
    QueueHandler. That call is O(microseconds) and does no disk I/O,
    so it cannot stall the 30 FPS render loop.
  - A single background thread (QueueListener) drains the queue
    and writes to a rotating file + the console. All formatting,
    flushing and disk syscalls happen there.
  - `print(...)` calls inside the existing scripts are NOT touched —
    they continue to go to stdout. The console log handler is set
    to WARNING so we don't double-print info messages.

Usage:
    from dwpose_engagement.applog import setup_logging
    log = setup_logging("engagement")
    log.info("starting up...")
"""

from __future__ import annotations

import atexit
import logging
import logging.handlers
import os
import platform
import queue
import sys
from datetime import datetime
from pathlib import Path

_LISTENER: logging.handlers.QueueListener | None = None
_LOG_PATH: Path | None = None


def _default_log_dir(script_name: str) -> Path:
    # Repo root = parent of src/. Falls back to CWD if the layout is unexpected.
    here = Path(__file__).resolve()
    for candidate in (here.parents[1], here.parents[2]):
        if (candidate / "src").is_dir():
            return candidate / "data" / "logs" / script_name
    return Path.cwd() / "data" / "logs" / script_name


def setup_logging(
    script_name: str,
    *,
    level: int = logging.INFO,
    console_level: int = logging.WARNING,
    log_dir: Path | None = None,
    extra_context: dict[str, str] | None = None,
) -> logging.Logger:
    """Configure the root logger once. Safe to call multiple times.

    Args:
        script_name: short identifier used in the log file name
            (e.g. "engagement", "emotibit", "subscriber").
        level: minimum level written to the log file.
        console_level: minimum level echoed to stderr. Default WARNING
            so existing print()s remain the primary on-screen output.
        log_dir: override directory (default: <repo>/data/logs/<script>).
        extra_context: optional dict of key/value pairs written to the
            log header (e.g. camera index, redis host).

    Returns:
        A logger named after the script. Modules should use
        `logging.getLogger(__name__)` rather than this returned logger.
    """
    global _LISTENER, _LOG_PATH

    root = logging.getLogger()
    # Second call in the same process: no-op so re-importing modules
    # doesn't add duplicate handlers.
    if _LISTENER is not None:
        return logging.getLogger(script_name)

    root.setLevel(min(level, console_level))

    log_dir = log_dir or _default_log_dir(script_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"{script_name}_{ts}_pid{os.getpid()}.log"
    _LOG_PATH = log_path

    # 10 MB per file, keep 5 rotations = ~50 MB ceiling per script.
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)s] "
            "%(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(
        fmt="%(levelname)s %(name)s: %(message)s",
    ))

    # The queue is the whole point: producer threads only touch this,
    # disk I/O happens on the listener thread.
    log_queue: queue.Queue = queue.Queue(-1)
    queue_handler = logging.handlers.QueueHandler(log_queue)
    root.addHandler(queue_handler)

    _LISTENER = logging.handlers.QueueListener(
        log_queue, file_handler, console_handler, respect_handler_level=True
    )
    _LISTENER.start()
    atexit.register(_shutdown)

    log = logging.getLogger(script_name)
    log.info("=" * 70)
    log.info("Logging started: %s", script_name)
    log.info("Log file: %s", log_path)
    log.info("Platform: %s %s | Python %s | PID %d",
             platform.system(), platform.release(),
             platform.python_version(), os.getpid())
    if extra_context:
        for k, v in extra_context.items():
            log.info("  %s=%s", k, v)
    log.info("=" * 70)
    return log


def get_log_path() -> Path | None:
    """Path of the active log file, or None before setup_logging() runs."""
    return _LOG_PATH


def _shutdown() -> None:
    global _LISTENER
    if _LISTENER is not None:
        try:
            _LISTENER.stop()
        except Exception:
            pass
        _LISTENER = None


def install_excepthook(log: logging.Logger | None = None) -> None:
    """Route uncaught exceptions into the log so post-mortem is possible."""
    log = log or logging.getLogger("uncaught")

    def _hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _hook
