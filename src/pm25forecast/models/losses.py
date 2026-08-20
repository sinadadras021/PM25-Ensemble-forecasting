r"""Custom loss functions (paper §4.4, Eqs. 19-20).

The paper trains its base learners on two purpose-built objectives instead of
plain MSE, for a reason worth stating plainly: a symmetric squared error trained
on a right-skewed pollutant series learns to predict the middle of the
distribution. It scores well on the many ordinary hours and systematically
under-shoots the few episodes that matter. Both losses below re-price that
trade-off, from opposite directions.

**Inverse-Weighted Loss (IWL), Eq. 19**

.. math::

    L_{IWL} = \frac{1}{N}\sum_n \left(1 + \frac{\alpha}{y_n + 1}\right)(y_n - \hat{y}_n)^2

The weight *falls* as the observation grows, so clean hours are emphasised. This
protects the low end from being swamped: without it, a model tuned only for
episodes drifts high everywhere and produces a stream of false alarms on ordinary
days.

**Exponential Extreme Loss (EEL), Eq. 20**

.. math::

    L_{EEL} = \frac{1}{N}\sum_n e^{y_n}\,\lvert y_n - \hat{y}_n \rvert^{\,p}

The weight *grows exponentially* with the observation, so heavy-pollution hours
dominate the gradient. This is the recall-oriented member of the pair.

Together the two objectives are what makes the four base learners genuinely
complementary rather than four seeds of one model — which is the precondition for
any of the six ensembles in §4.2 to add value.

Scale matters and is easy to get wrong
--------------------------------------
Both losses are applied in **scaled target space**, where ``y`` lies in ``[0, 1]``.
In EEL that bounds the weight at ``e^1 ≈ 2.72``. Applying the same formula to raw
µg/m³ would give ``e^300``, which overflows to ``inf`` and destroys the run in a
single step. :func:`resolve_loss` therefore takes the scaling contract as an
explicit argument rather than trusting the caller to remember.

Every loss masks NaN labels: the target is not fully observed, and a horizon with
a few missing hours should still teach the model from the hours it has.
"""

from __future__ import annotations

from typing import Dict

import tensorflow as tf

__all__ = [
    "InverseWeightedLoss",
    "ExponentialExtremeLoss",
    "MaskedMSE",
    "DEFAULT_LOSS_BY_MODEL",
    "resolve_loss",
]


def _mask_and_reduce(weighted: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Mean over observed labels only; an all-missing window contributes 0."""
    axes = tf.range(1, tf.rank(weighted))
    numerator = tf.reduce_sum(weighted * mask, axis=axes)
    count = tf.reduce_sum(mask, axis=axes)
    return tf.math.divide_no_nan(numerator, count)


def _prepare(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    observed = tf.math.is_finite(y_true)
    safe_true = tf.where(observed, y_true, tf.zeros_like(y_true))
    return safe_true, y_pred, tf.cast(observed, tf.float32)


@tf.keras.utils.register_keras_serializable(package="pm25forecast")
class InverseWeightedLoss(tf.keras.losses.Loss):
    r"""Eq. 19 -- squared error weighted by :math:`1 + \alpha / (y + 1)`.

    Parameters
    ----------
    alpha:
        Weighting intensity. At ``alpha = 10`` a clean hour (``y = 0`` scaled)
        carries weight 11 while a peak hour (``y = 1``) carries weight 6 -- a
        factor of about two in favour of the low end.

    Examples
    --------
    A perfect forecast costs nothing, and a missing label is ignored entirely:

    >>> loss = InverseWeightedLoss(alpha=10.0)
    >>> y = tf.constant([[0.2, float("nan"), 0.8]])
    >>> float(loss(y, tf.constant([[0.2, 99.0, 0.8]]))) < 1e-6
    True

    The same absolute error is penalised more at low concentration:

    >>> low = float(loss(tf.constant([[0.0]]), tf.constant([[0.1]])))
    >>> high = float(loss(tf.constant([[1.0]]), tf.constant([[1.1]])))
    >>> low > high
    True
    """

    def __init__(self, alpha: float = 10.0, name: str = "inverse_weighted_loss", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        if alpha < 0:
            raise ValueError(f"IWL alpha must be >= 0, got {alpha}.")
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        safe_true, y_pred, mask = _prepare(y_true, y_pred)
        weight = 1.0 + self.alpha / (tf.abs(safe_true) + 1.0)
        return _mask_and_reduce(weight * tf.square(safe_true - y_pred), mask)

    def get_config(self) -> Dict[str, object]:
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config


@tf.keras.utils.register_keras_serializable(package="pm25forecast")
class ExponentialExtremeLoss(tf.keras.losses.Loss):
    r"""Eq. 20 -- :math:`e^{y}\,\lvert y - \hat{y}\rvert^{p}`.

    Parameters
    ----------
    power:
        Exponent on the residual. ``p = 2`` recovers a weighted squared error;
        ``p = 1`` a weighted absolute error.
    max_exponent:
        Safety clamp on the exponent before ``exp``. With scaled targets the
        argument never exceeds 1, so the clamp is inert — it exists so that
        accidentally passing raw µg/m³ produces a bounded loss and a diagnosable
        run rather than an immediate ``inf`` and a wall of NaN gradients.

    Examples
    --------
    An error at a peak costs far more than the same error at a clean hour:

    >>> loss = ExponentialExtremeLoss(power=2.0)
    >>> high = float(loss(tf.constant([[1.0]]), tf.constant([[0.9]])))
    >>> low = float(loss(tf.constant([[0.0]]), tf.constant([[0.1]])))
    >>> round(high / low, 3)
    2.718
    """

    def __init__(
        self,
        power: float = 2.0,
        max_exponent: float = 10.0,
        name: str = "exponential_extreme_loss",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if power <= 0:
            raise ValueError(f"EEL power must be > 0, got {power}.")
        self.power = float(power)
        self.max_exponent = float(max_exponent)

    def call(self, y_true, y_pred):
        safe_true, y_pred, mask = _prepare(y_true, y_pred)
        exponent = tf.clip_by_value(tf.abs(safe_true), 0.0, self.max_exponent)
        weight = tf.exp(exponent)
        residual = tf.pow(tf.abs(safe_true - y_pred) + tf.keras.backend.epsilon(), self.power)
        return _mask_and_reduce(weight * residual, mask)

    def get_config(self) -> Dict[str, object]:
        config = super().get_config()
        config.update({"power": self.power, "max_exponent": self.max_exponent})
        return config


@tf.keras.utils.register_keras_serializable(package="pm25forecast")
class MaskedMSE(tf.keras.losses.Loss):
    """Plain squared error over observed labels -- used by the MLP meta-learner
    (paper Table 2)."""

    def __init__(self, name: str = "masked_mse", **kwargs) -> None:
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        safe_true, y_pred, mask = _prepare(y_true, y_pred)
        return _mask_and_reduce(tf.square(safe_true - y_pred), mask)


#: Objective per architecture (paper Table 2, "Loss Function" row).
DEFAULT_LOSS_BY_MODEL: Dict[str, str] = {
    "lstm": "iwl",
    "gru": "iwl",
    "bi_lstm": "eel",
    "bi_gru": "eel",
}


def resolve_loss(
    spec: str, *, alpha: float = 10.0, power: float = 2.0, targets_are_scaled: bool = True
) -> tf.keras.losses.Loss:
    """Instantiate a loss by short name.

    Parameters
    ----------
    targets_are_scaled:
        Must be ``True`` for the EEL. Passing ``False`` raises rather than
        silently building a loss that overflows on the first batch -- the failure
        would otherwise appear as unexplained NaN weights several epochs later.

    >>> isinstance(resolve_loss("iwl"), InverseWeightedLoss)
    True
    """
    if spec in {"iwl", "inverse_weighted"}:
        return InverseWeightedLoss(alpha=alpha)
    if spec in {"eel", "exponential_extreme"}:
        if not targets_are_scaled:
            raise ValueError(
                "The exponential extreme loss requires targets scaled to [0, 1]. On raw "
                "micrograms per cubic metre, exp(y) overflows to inf and every gradient "
                "becomes NaN. Scale the target first, or choose a different loss."
            )
        return ExponentialExtremeLoss(power=power)
    if spec in {"mse", "masked_mse"}:
        return MaskedMSE()
    raise ValueError(f"Unknown loss {spec!r}; known: iwl, eel, mse.")
