"""Reproducibility: the same seed must give the same numbers.

A deep-learning result that changes between runs of identical code is an
anecdote, not evidence. ``seed: 42`` in the config is only a promise; this file
checks that the promise is kept.
"""

from __future__ import annotations

import numpy as np
import pytest

from pm25forecast.seeding import environment_fingerprint, set_global_seed

tf = pytest.importorskip("tensorflow")


def _train_once() -> np.ndarray:
    """Seed, build, fit and predict -- the whole cycle, from scratch."""
    set_global_seed(42, deterministic=True)
    tf.keras.backend.clear_session()

    from pm25forecast.models.base import build_base_model

    rng = np.random.default_rng(0)  # the data is fixed, so only the model varies
    X = rng.normal(size=(48, 48, 6)).astype("float32")
    y = rng.normal(size=(48, 48)).astype("float32")

    model = build_base_model(
        "lstm", input_steps=48, n_features=6, horizon=48,
        rnn_units=(16, 8), bi_units=(8, 8), dropout=0.3,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(X, y, epochs=2, batch_size=8, verbose=0, shuffle=True)
    return model.predict(X, verbose=0)


def test_same_seed_gives_bit_identical_predictions():
    """Two independent runs of identical code must not differ at all.

    Not "close to" -- identical. Weight initialisation, batch shuffling and
    dropout masks are all sources of randomness here, and ``set_global_seed`` is
    expected to pin every one of them. A tolerance-based check would pass even if
    one of them had been left unseeded, which is exactly the failure worth
    catching.
    """
    first = _train_once()
    second = _train_once()

    assert np.array_equal(first, second), (
        "identical code and identical seed produced different predictions; "
        f"largest difference {float(np.max(np.abs(first - second)))}"
    )


def test_environment_is_recorded_alongside_results():
    """Determinism holds for one (hardware, library-version) triple.

    So the triple has to be archived next to the numbers, or the reproducibility
    claim cannot be checked by anyone else.
    """
    info = environment_fingerprint()
    for key in ("python", "platform", "numpy", "pandas"):
        assert info.get(key), f"environment fingerprint is missing {key!r}"
