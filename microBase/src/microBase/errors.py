"""Exception hierarchy shared by microBase and its consumers.

Library code raises these; only the CLI entry points / GUI boundaries decide
how to report them (log + exit code / dialog). Never call ``sys.exit()`` from
library code — a GUI, worker thread or data-loader worker must be able to
catch the failure.
"""

from __future__ import annotations


class MicroMaxError(Exception):
    """Base class for every error the suite raises deliberately."""


class ConfigError(MicroMaxError):
    """Invalid or incomplete configuration (bad keys, missing required values)."""


class DatasetError(MicroMaxError):
    """Dataset layout/metadata problems (bad regex, missing columns, breakage
    the caller cannot quarantine away)."""


class ImageReadError(MicroMaxError):
    """A file referenced by the metadata is missing or cannot be decoded.

    ``path`` is the offending file when known, else None. Kept as a distinct
    class so pipeline callers can quarantine a broken row instead of aborting
    the whole run.
    """

    def __init__(self, path, message):
        super().__init__(message)
        self.path = path


class DataError(MicroMaxError):
    """Persisted-data problems (corrupt/missing DB, unexpected schema)."""


class DependencyError(MicroMaxError):
    """A required optional dependency or model artifact is missing."""
