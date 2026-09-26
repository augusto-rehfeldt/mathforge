"""Process helpers: a .env loader (no python-dotenv dependency) and a Ctrl-C that stops at once."""

import os
import signal
import threading
from pathlib import Path


def load_local_env(env_path=None) -> bool:
    """Load KEY=value lines from .env; existing environment values always win."""
    path = Path(env_path) if env_path is not None else Path(__file__).resolve().parent.parent / ".env"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return False
    for line in lines:
        key, sep, value = line.strip().partition("=")
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if sep and key.strip() and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value)
    return True


_console_handler = None  # the ctypes callback must outlive the call that registers it


def exit_on_ctrl_c(save=None, message: str = "Stopped.") -> None:
    """Make Ctrl-C save and exit immediately instead of after the in-flight AI call.

    Python only raises KeyboardInterrupt once the main thread gets back to
    bytecode, and on Windows a blocking socket read (a minutes-long model
    request) never does. The console handler below runs on its own thread, so it
    fires at once. The files it saves are written atomically; a second Ctrl-C
    skips the save and exits straight away.
    """
    first = threading.Lock()

    def stop(*_):
        if first.acquire(blocking=False) and save:
            try:
                save()
            except Exception as exc:  # stopping matters more than one failed save
                print(f"\nsave on exit failed: {exc}", flush=True)
        print(f"\n{message}", flush=True)
        os._exit(130)

    if os.name != "nt":
        signal.signal(signal.SIGINT, stop)
        return
    import ctypes

    global _console_handler

    def on_console_event(event):
        if event in (0, 1):  # CTRL_C_EVENT, CTRL_BREAK_EVENT
            stop()
        return 0  # anything else (window close, logoff): let the default run

    _console_handler = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(on_console_event)
    ctypes.windll.kernel32.SetConsoleCtrlHandler(_console_handler, True)
