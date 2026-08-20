"""Reproducibility controls.

Deep-learning results that cannot be reproduced are anecdotes, not evidence.
:func:`set_global_seed` pins every source of randomness this project touches and
optionally switches TensorFlow into deterministic-kernel mode.

Caveat worth stating in any paper that uses this: determinism holds for a fixed
(hardware, driver, library-version) triple. cuDNN picks different reduction
orders on different GPUs, so bit-identical results across machines are not
guaranteed even with ``deterministic=True``. Run :func:`environment_fingerprint`
and archive its output alongside your results.
"""

from __future__ import annotations

import os
import platform
import random
from typing import Any, Dict

import numpy as np


def set_global_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """Seed ``random``, ``numpy`` and TensorFlow, and configure GPU memory growth.

    Must be called *before* TensorFlow builds any graph.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
    except ImportError:  # pragma: no cover - TF is optional for pure-data work
        return

    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:  # pragma: no cover - older TF releases
            pass

    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:  # pragma: no cover - already initialised
            pass


def environment_fingerprint() -> Dict[str, Any]:
    """Capture everything a reader needs to reproduce a run.

    Archive this next to your metrics. It is the difference between
    "we got R2 = 0.91" and "we got R2 = 0.91, here is exactly how".
    """
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
    }
    for module_name, key in (
        ("pandas", "pandas"),
        ("sklearn", "scikit_learn"),
        ("tensorflow", "tensorflow"),
        ("scipy", "scipy"),
    ):
        try:
            module = __import__(module_name)
            info[key] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[key] = None

    try:
        import tensorflow as tf

        info["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except ImportError:
        info["gpus"] = None
    return info
