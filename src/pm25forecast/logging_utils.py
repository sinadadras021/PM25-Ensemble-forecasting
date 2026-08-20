"""Consistent, machine-parsable logging for every stage of the pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_file: Optional[str | Path] = None,
    force: bool = False,
) -> None:
    """Install a stream handler (and optionally a file handler) exactly once.

    Calling this repeatedly is safe: subsequent calls are no-ops unless
    ``force=True``. Library code should never call it -- only entry points.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(level)

    # TensorFlow and absl are extremely chatty at import time.
    for noisy in ("tensorflow", "absl", "h5py", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Use ``get_logger(__name__)``."""
    return logging.getLogger(name)


def section(logger: logging.Logger, title: str, width: int = 78) -> None:
    """Emit a visually scannable section header."""
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)
