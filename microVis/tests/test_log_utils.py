"""microVis logging setup: noisy third-party loggers stay quiet."""

import logging


def test_noisy_third_party_loggers_are_quieted():
    """fontTools' per-step PDF subsetting INFO must not flood the log."""
    from microVis.log_utils import setup_logging

    setup_logging()
    for name in ("fontTools", "fontTools.subset",
                 "matplotlib.font_manager", "PIL"):
        assert not logging.getLogger(name).isEnabledFor(logging.INFO), name
