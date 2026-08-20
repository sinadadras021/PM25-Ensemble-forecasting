"""The four base learners (paper §4.1, Table 2).

All four obey one contract::

    input  (batch, 48, n_features)  ->  output (batch, 48)

so they are interchangeable and comparable on identical windows, and the six
ensembles of §4.2 can treat their outputs as a uniform 48-dimensional feature
block.

Two families, differing in *both* structure and objective (Table 2):

===================  ==================  =====================================
family               structure           loss
===================  ==================  =====================================
``lstm`` / ``gru``   64 -> 32, ReLU      inverse-weighted (IWL, Eq. 19)
``bi_lstm``/``bi_gru``  2 x 32, ReLU     exponential extreme (EEL, Eq. 20)
===================  ==================  =====================================

Pairing a *deeper, narrowing* unidirectional stack with the low-end-emphasising
IWL, and a *shallower, wider* bidirectional stack with the peak-emphasising EEL,
is what produces genuinely decorrelated errors. Four models with the same shape
and the same objective would differ only by random seed, their mistakes would
coincide, and no meta-learner could recover anything from combining them.

Bidirectionality is not leakage: the encoder reads the 48-hour *input* window,
which is fully observed at issue time.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    GRU,
    Input,
    LSTM,
)
from tensorflow.keras.optimizers import Adam

from . import losses as loss_lib

__all__ = [
    "AVAILABLE_BASE_MODELS",
    "build_base_model",
    "compile_base_model",
    "register_model",
]

_REGISTRY: Dict[str, Callable[..., Model]] = {}


def register_model(name: str):
    def wrapper(fn: Callable[..., Model]) -> Callable[..., Model]:
        _REGISTRY[name] = fn
        return fn

    return wrapper


# --------------------------------------------------------------------------- #
def _stacked_rnn(
    cell,
    *,
    input_steps: int,
    n_features: int,
    horizon: int,
    units: Tuple[int, int],
    dropout: float,
    bidirectional: bool,
    name: str,
) -> Model:
    """Shared skeleton: two recurrent layers, then a dense head emitting the
    whole 48-step trajectory at once.

    A single dense output layer over the flattened final state -- rather than a
    ``RepeatVector`` decoder -- is what Table 2 specifies, and it suits a horizon
    this long: an autoregressive decoder unrolled 48 steps accumulates its own
    error, whereas a direct multi-output head predicts each lead time from the
    same encoded state and degrades more gracefully.
    """
    first = cell(units[0], return_sequences=True)
    second = cell(units[1], return_sequences=False)
    if bidirectional:
        first, second = Bidirectional(first), Bidirectional(second)

    return Sequential(
        [
            Input(shape=(input_steps, n_features)),
            first,
            Dropout(dropout),
            second,
            Dropout(dropout),
            Dense(64, activation="relu"),
            Dense(horizon, activation="linear"),
        ],
        name=name,
    )


@register_model("lstm")
def _lstm(input_steps, n_features, horizon, rnn_units, bi_units, dropout) -> Model:
    """Unidirectional LSTM, 64 -> 32 (Table 2, "RNNs")."""
    return _stacked_rnn(
        LSTM, input_steps=input_steps, n_features=n_features, horizon=horizon,
        units=rnn_units, dropout=dropout, bidirectional=False, name="LSTM",
    )


@register_model("gru")
def _gru(input_steps, n_features, horizon, rnn_units, bi_units, dropout) -> Model:
    """Unidirectional GRU, 64 -> 32. Fewer gates than LSTM, which suits the
    modest effective sample size of overlapping daily windows."""
    return _stacked_rnn(
        GRU, input_steps=input_steps, n_features=n_features, horizon=horizon,
        units=rnn_units, dropout=dropout, bidirectional=False, name="GRU",
    )


@register_model("bi_lstm")
def _bi_lstm(input_steps, n_features, horizon, rnn_units, bi_units, dropout) -> Model:
    """Bidirectional LSTM, 2 x 32 (Table 2, "Bi-RNNs")."""
    return _stacked_rnn(
        LSTM, input_steps=input_steps, n_features=n_features, horizon=horizon,
        units=bi_units, dropout=dropout, bidirectional=True, name="Bi_LSTM",
    )


@register_model("bi_gru")
def _bi_gru(input_steps, n_features, horizon, rnn_units, bi_units, dropout) -> Model:
    """Bidirectional GRU, 2 x 32."""
    return _stacked_rnn(
        GRU, input_steps=input_steps, n_features=n_features, horizon=horizon,
        units=bi_units, dropout=dropout, bidirectional=True, name="Bi_GRU",
    )


AVAILABLE_BASE_MODELS = tuple(sorted(_REGISTRY))


# --------------------------------------------------------------------------- #
def build_base_model(
    name: str,
    *,
    input_steps: int,
    n_features: int,
    horizon: int,
    rnn_units: Tuple[int, int] = (64, 32),
    bi_units: Tuple[int, int] = (32, 32),
    dropout: float = 0.20,
) -> Model:
    """Instantiate an uncompiled architecture from the registry."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown base model {name!r}; available: {AVAILABLE_BASE_MODELS}")
    for label, value in (("input_steps", input_steps), ("n_features", n_features),
                         ("horizon", horizon)):
        if value < 1:
            raise ValueError(f"build_base_model({name!r}): {label} must be >= 1, got {value}.")
    return _REGISTRY[name](
        input_steps, n_features, horizon, tuple(rnn_units), tuple(bi_units), dropout
    )


def compile_base_model(
    model: Model,
    *,
    name: str,
    learning_rate: float = 1e-3,
    loss: Optional[str] = None,
    iwl_alpha: float = 10.0,
    eel_power: float = 2.0,
) -> Model:
    """Attach Adam and the architecture's Table 2 objective.

    The loss is chosen by architecture unless overridden, so the IWL/EEL pairing
    that gives the ensemble its diversity cannot be lost by accident.
    """
    spec = loss or loss_lib.DEFAULT_LOSS_BY_MODEL.get(name, "mse")
    loss_fn = loss_lib.resolve_loss(
        spec, alpha=iwl_alpha, power=eel_power, targets_are_scaled=True
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_fn)
    return model
