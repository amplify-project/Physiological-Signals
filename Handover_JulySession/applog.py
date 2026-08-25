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
_CONSOLE_MIRRORED: bool = False


def _default_log_dir(script_name: str) -> Path:
    # Handover bundle is self-contained: always anchor logs to this file's
    # parent folder (Handover_JuneSession/data/logs/<script>) so log location
    # doesn't depend on the user's CWD when launching the .bat files.
    here = Path(__file__).resolve()
    return here.parent / "data" / "logs" / script_name


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


class _TeeStream:
    """Write-through stream wrapper that also mirrors whole lines to a logger.

    The wrapped stream still receives every write, so the terminal looks
    unchanged; completed lines are additionally forwarded to `log_fn` so
    console output (boot banner, camera negotiation, warnings) is preserved
    in the session log. Only newline-terminated lines are forwarded to avoid
    fragmenting messages across partial writes."""

    def __init__(self, stream, log_fn):
        self._stream = stream
        self._log_fn = log_fn
        self._buf = ""

    def write(self, text):
        self._stream.write(text)
        self._buf += text
        if "\n" in self._buf:
            *lines, self._buf = self._buf.split("\n")
            for line in lines:
                line = line.rstrip("\r")
                if line.strip():
                    try:
                        self._log_fn(line)
                    except Exception:
                        pass

    def flush(self):
        self._stream.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def mirror_console_to_log(level: int = logging.INFO) -> None:
    """Tee stdout/stderr into the log file so on-screen output is recorded.

    Call once, right after setup_logging(). print()s keep appearing on screen
    and are additionally captured in the session log so a run can be reviewed
    afterwards. The console handler captured the real stderr at setup time, so
    wrapping here does not loop. No-op before setup or if called twice."""
    global _CONSOLE_MIRRORED
    if _CONSOLE_MIRRORED or _LISTENER is None:
        return
    out_log = logging.getLogger("stdout")
    err_log = logging.getLogger("stderr")
    sys.stdout = _TeeStream(sys.stdout, lambda m: out_log.log(level, m))
    sys.stderr = _TeeStream(sys.stderr, lambda m: err_log.log(level, m))
    _CONSOLE_MIRRORED = True


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
