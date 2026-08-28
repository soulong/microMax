from __future__ import annotations

import logging
import sys

from microProfiler.log_utils import _ensure_std_streams


def _alloc_console() -> None:
    """Allocate a console window if running without one (pythonw.exe)."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        if kernel32.GetConsoleWindow() == 0:
            kernel32.AllocConsole()
            sys.stdout = open("CONOUT$", "w", encoding="utf-8")
            sys.stderr = open("CONOUT$", "w", encoding="utf-8")
    except Exception:
        pass


def main() -> int:
    """Dispatch to GUI or CLI based on command-line arguments."""
    _ensure_std_streams()
    debug_mode = "--debug" in sys.argv

    if "--version" in sys.argv:
        from importlib.metadata import version
        print(f"microProfiler {version('microProfiler')}")
        return 0

    if "--help" in sys.argv and "run" not in sys.argv:
        from microProfiler.cli import build_parser
        build_parser().print_help()
        return 0

    if len(sys.argv) == 1 or (len(sys.argv) == 2 and debug_mode):
        if debug_mode:
            from microProfiler.log_utils import set_default_logging_level
            set_default_logging_level(logging.DEBUG)
        _alloc_console()
        from microProfiler.gui.app import main as gui_main
        gui_main()
        return 0

    if "run" in sys.argv:
        from microProfiler.cli import main as cli_main
        return cli_main(sys.argv[1:])

    print(f"Unknown argument: {sys.argv[1]}", file=sys.stderr)
    print("Usage: microprofiler [run | --version | --help]", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
