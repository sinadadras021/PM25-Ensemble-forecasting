"""Figure 2 architectures, Table 2 MSE and Table 3 callbacks; no fit at import."""

import random
from functools import lru_cache
import numpy as np


def set_seed(seed):
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


@lru_cache(maxsize=1)
def _observed_mse_class():
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="pm25forecast")
    class ObservedMSE(tf.keras.losses.Loss):
        """Ordinary MSE over observed labels; missing targets are not imputed."""

        def call(self, y_true, y_pred):
            finite = tf.math.is_finite(y_true)
            safe = tf.where(finite, y_true, tf.zeros_like(y_true))
            squares = tf.where(finite, tf.square(safe - y_pred), tf.zeros_like(y_pred))
            return tf.math.divide_no_nan(
                tf.reduce_sum(squares), tf.reduce_sum(tf.cast(finite, y_pred.dtype))
            )

    return ObservedMSE


def observed_mse():
    """Return a fresh stateless loss without registering its class twice."""
    return _observed_mse_class()(name="observed_mse")


def callbacks(cfg):
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ReduceLROnPlateau,
        TerminateOnNaN,
    )

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_learning_rate,
        ),
        TerminateOnNaN(),
    ]


def build_base(name, n_features, cfg):
    """Construct the recurrent sequence architecture drawn in manuscript Figure 2."""
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        Input,
        GRU,
        LSTM,
        Bidirectional,
        Dropout,
        TimeDistributed,
        Dense,
    )

    tf.keras.backend.clear_session()
    set_seed(cfg.seed)
    cell = GRU if "gru" in name else LSTM
    if name not in {"bi_gru", "bi_lstm", "gru", "lstm"}:
        raise ValueError(name)
    first = cell(32 if name.startswith("bi_") else 64, return_sequences=True)
    second = cell(32, return_sequences=True)
    if name.startswith("bi_"):
        first = Bidirectional(first)
        second = Bidirectional(second)
    # ReLU belongs to the Dense(32) head, as explicitly drawn in Figure 2.
    model = Sequential(
        [
            Input(shape=(48, n_features)),
            first,
            Dropout(cfg.dropout),
            second,
            Dropout(cfg.dropout),
            TimeDistributed(Dense(32, activation="relu")),
            TimeDistributed(Dense(1, activation="linear")),
        ],
        name=name,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=observed_mse(),
    )
    return model


def build_meta(n_features, cfg):
    """Build the 32/16/8-unit MLP with four separate base forecast inputs."""
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, Dense

    set_seed(cfg.seed)
    model = Sequential(
        [
            Input(shape=(n_features,)),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(1, activation="linear"),
        ],
        name="MLP_Meta",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate), loss="mse"
    )
    return model
