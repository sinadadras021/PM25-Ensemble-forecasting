r"""The six ensembles, in three strategies (paper §4.2, Eqs. 6-13, Table 2).

Each strategy embodies a different theory of *why* combining models helps, and
the paper's contribution is comparing them on the same footing rather than
asserting one:

============================  ====================================================
strategy                      claim being tested
============================  ====================================================
Statistical Averaging         members are individually noisy but jointly unbiased,
(``sa``, ``pwe``)             so a fixed linear blend cancels variance
Non-Linear Meta-Learning      member error depends on atmospheric state in a way a
(``mlp_meta``, ``xgb_meta``)  linear blend cannot express
Context-Aware Selection       members are *specialists*: rather than averaging
(``h_mos``, ``sms``)          them, identify which one to trust right now
============================  ====================================================

Every ensemble here implements the same tiny interface -- ``fit(...)`` then
``predict(...)`` -- so the pipeline treats them uniformly and adding a seventh is
a self-contained class.

**Fitting discipline.** Every ensemble is fitted on the *validation* block, where
the base members' predictions are genuinely out-of-sample, and is then applied
once to the test year. Fitting a stack on in-sample base predictions is the
classic failure of stacked generalisation: the meta-learner learns to trust
whichever member memorised the training set hardest, and the whole structure
degrades exactly where it was supposed to help.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "AVAILABLE_ENSEMBLES",
    "ENSEMBLE_STRATEGY",
    "SimpleAveraging",
    "PerformanceWeightedEnsemble",
    "MLPMetaLearner",
    "XGBoostMetaLearner",
    "HourlyMOS",
    "SmartModelSelector",
    "build_ensemble",
]

AVAILABLE_ENSEMBLES = ("sa", "pwe", "mlp_meta", "xgb_meta", "h_mos", "sms")

#: Which of the paper's three strategies each ensemble belongs to (§4.2).
ENSEMBLE_STRATEGY: Dict[str, str] = {
    "sa": "Statistical Averaging",
    "pwe": "Statistical Averaging",
    "mlp_meta": "Non-Linear Meta-Learning",
    "xgb_meta": "Non-Linear Meta-Learning",
    "h_mos": "Context-Aware Selection",
    "sms": "Context-Aware Selection",
}


def _flatten(members: np.ndarray) -> np.ndarray:
    """``(n, horizon, m)`` -> ``(n * horizon, m)``, lead-major within each window."""
    n, horizon, m = members.shape
    return members.reshape(n * horizon, m)


def _restore(flat: np.ndarray, n: int, horizon: int) -> np.ndarray:
    return np.asarray(flat, dtype=np.float64).reshape(n, horizon)


# =========================================================================== #
# 4.2.1  Statistical Averaging
# =========================================================================== #
class SimpleAveraging:
    r"""**SA** -- uniform mean of the members (paper Eq. 6).

    .. math::  y_t^{SA} = \frac{1}{T}\sum_{i=1}^{T} y_{t,i}

    Assumes equal reliability. It has no parameters, so it cannot overfit, and
    that is precisely its role: it is the control every other ensemble must beat
    to justify its extra machinery. An elaborate meta-learner that ties with SA
    has demonstrated nothing.
    """

    name = "sa"

    def fit(self, members: np.ndarray, y: np.ndarray, context: Optional[np.ndarray] = None):
        return self

    def predict(self, members: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        return np.asarray(members, dtype=np.float64).mean(axis=-1)


class PerformanceWeightedEnsemble:
    r"""**PWE** -- convex weights optimised by SLSQP (paper Eqs. 7-8).

    .. math::

        \min_{w} \sum_n \Big(y^{(n)} - \sum_i w_i y_i^{(n)}\Big)^2
        \quad\text{s.t.}\quad \sum_i w_i = 1,\ 0 \le w_i \le 1

    The constraints are what separate this from ordinary least squares, and they
    matter. Unconstrained regression on correlated member forecasts routinely
    returns large positive and negative coefficients that cancel -- a fit that
    looks excellent in-sample and inverts catastrophically when the correlation
    structure shifts. Restricting to a convex combination keeps the prediction
    inside the members' own envelope, so PWE can never be worse than the worst
    member by more than the members disagree.
    """

    name = "pwe"

    def __init__(self, method: str = "SLSQP", max_iter: int = 1000) -> None:
        self.method = method
        self.max_iter = max_iter
        self.weights_: Optional[np.ndarray] = None

    def fit(self, members: np.ndarray, y: np.ndarray, context: Optional[np.ndarray] = None):
        from scipy.optimize import minimize

        X = _flatten(np.asarray(members, dtype=np.float64))
        target = np.asarray(y, dtype=np.float64).ravel()
        mask = np.isfinite(target) & np.isfinite(X).all(axis=1)
        X, target = X[mask], target[mask]
        if X.size == 0:
            raise ValueError("PWE.fit: no complete (members, target) pairs.")

        n_members = X.shape[1]
        start = np.full(n_members, 1.0 / n_members)
        n_rows = X.shape[0]

        # The objective is the *mean* squared residual, not the sum, and the
        # analytic gradient is supplied. Both matter in practice: on a real
        # station the sum runs to 10^6-10^8, SLSQP's default convergence
        # tolerance is then satisfied at the starting point, and the optimiser
        # returns `success=True` having moved nowhere -- leaving PWE
        # indistinguishable from simple averaging while reporting no error at all.
        def objective(w: np.ndarray) -> float:
            residual = target - X @ w
            return float(residual @ residual / n_rows)

        def gradient(w: np.ndarray) -> np.ndarray:
            residual = target - X @ w
            return -2.0 * (X.T @ residual) / n_rows

        result = minimize(
            objective,
            start,
            jac=gradient,
            method=self.method,
            bounds=[(0.0, 1.0)] * n_members,
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda w: np.sum(w) - 1.0,
                    "jac": lambda w: np.ones_like(w),
                }
            ],
            options={"maxiter": self.max_iter, "ftol": 1e-10},
        )
        weights = np.clip(result.x, 0.0, 1.0)
        total = weights.sum()
        if not result.success or total <= 0 or not np.isfinite(weights).all():
            logger.warning(
                "PWE: %s did not converge (%s); falling back to uniform weights, which makes "
                "this ensemble identical to SA. Treat its row in the results as such.",
                self.method, getattr(result, "message", "unknown"),
            )
            self.weights_ = start
        else:
            self.weights_ = weights / total
            improvement = 1.0 - np.sqrt(objective(self.weights_)) / np.sqrt(objective(start))
            logger.debug(
                "PWE: optimisation reduced RMSE by %.2f%% versus uniform weights.",
                improvement * 100,
            )
        logger.info(
            "PWE weights: %s",
            " | ".join(f"{w * 100:.1f}%" for w in self.weights_),
        )
        return self

    def predict(self, members: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("PWE.predict called before fit.")
        return np.tensordot(np.asarray(members, dtype=np.float64), self.weights_, axes=([2], [0]))


# =========================================================================== #
# 4.2.2  Non-Linear Meta-Learning
# =========================================================================== #
class MLPMetaLearner:
    r"""**MLP-Meta** -- feed-forward network over members + context (paper Eq. 9).

    Input at each forecast hour is the concatenation of the four member
    predictions, the concurrent meteorology, the cyclical time features and
    ``Hour_offset``. The last of these is what lets the network model *lead-time
    decay*: it can learn to lean on a different member at hour 40 than at hour 2,
    which no fixed linear blend can express.
    """

    name = "mlp_meta"

    def __init__(
        self,
        hidden: Sequence[int] = (32, 16, 8),
        dropout: float = 0.10,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10,
        seed: int = 42,
    ) -> None:
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.model_ = None
        self.scaler_ = None

    def _design(self, members: np.ndarray, context: Optional[np.ndarray]) -> np.ndarray:
        X = _flatten(np.asarray(members, dtype=np.float64))
        if context is not None:
            ctx = np.asarray(context, dtype=np.float64)
            X = np.hstack([X, ctx.reshape(X.shape[0], -1)])
        return X

    def fit(self, members: np.ndarray, y: np.ndarray, context: Optional[np.ndarray] = None):
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam

        X = self._design(members, context)
        target = np.asarray(y, dtype=np.float64).ravel()
        mask = np.isfinite(target) & np.isfinite(X).all(axis=1)
        X, target = X[mask], target[mask]
        if X.size == 0:
            raise ValueError("MLPMetaLearner.fit: no complete rows.")

        # Member predictions are in ug/m3 while cyclical features live in
        # [-1, 1]; without standardisation the first hidden layer would be
        # dominated by scale rather than by information.
        self.scaler_ = StandardScaler().fit(X)
        X = self.scaler_.transform(X)

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.seed)

        layers = [Input(shape=(X.shape[1],))]
        for width in self.hidden:
            layers += [Dense(width, activation="relu"), Dropout(self.dropout)]
        layers.append(Dense(1, activation="linear"))
        self.model_ = Sequential(layers, name="MLP_Meta")
        self.model_.compile(optimizer=Adam(self.learning_rate), loss="mse")

        # A chronological tail is held out for early stopping: the rows are
        # consecutive forecast hours, so a random split would place near-copies
        # on both sides and stopping would fire far too late.
        cut = max(int(len(X) * 0.9), 1)
        self.model_.fit(
            X[:cut], target[:cut],
            validation_data=(X[cut:], target[cut:]) if cut < len(X) else None,
            epochs=self.epochs, batch_size=self.batch_size, verbose=0,
            callbacks=[
                EarlyStopping(monitor="val_loss" if cut < len(X) else "loss",
                              patience=self.patience, restore_best_weights=True)
            ],
        )
        return self

    def predict(self, members: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("MLPMetaLearner.predict called before fit.")
        n, horizon = members.shape[0], members.shape[1]
        X = self.scaler_.transform(np.nan_to_num(self._design(members, context), nan=0.0))
        return _restore(self.model_.predict(X, verbose=0).ravel(), n, horizon)


class XGBoostMetaLearner:
    r"""**XGB-Meta** -- gradient-boosted trees over the same design (paper Eq. 10).

    Where the MLP fits a smooth function, boosted trees fit *thresholds*, which
    is a better match for how pollution episodes actually behave: below a certain
    wind speed the boundary layer collapses and the members' relationship to the
    truth changes regime. A tree can represent that break exactly; a small MLP
    smooths across it.

    Its feature importances also make the ensemble legible -- the paper uses them
    to say which atmospheric variables drive the correction.
    """

    name = "xgb_meta"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.seed = seed
        self.model_ = None
        self.feature_names_: List[str] = []

    def _design(self, members: np.ndarray, context: Optional[np.ndarray]) -> np.ndarray:
        X = _flatten(np.asarray(members, dtype=np.float64))
        if context is not None:
            X = np.hstack([X, np.asarray(context, dtype=np.float64).reshape(X.shape[0], -1)])
        return X

    def fit(
        self,
        members: np.ndarray,
        y: np.ndarray,
        context: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
    ):
        X = self._design(members, context)
        target = np.asarray(y, dtype=np.float64).ravel()
        mask = np.isfinite(target) & np.isfinite(X).all(axis=1)
        X, target = X[mask], target[mask]
        if X.size == 0:
            raise ValueError("XGBoostMetaLearner.fit: no complete rows.")
        self.feature_names_ = (
            list(feature_names) if feature_names else [f"f{i}" for i in range(X.shape[1])]
        )

        try:
            from xgboost import XGBRegressor

            self.model_ = XGBRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, subsample=self.subsample,
                objective="reg:squarederror", random_state=self.seed, n_jobs=-1,
                tree_method="hist",
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor

            logger.warning(
                "xgboost is not installed; falling back to sklearn GradientBoostingRegressor "
                "with matched hyperparameters. Results will be close but not identical -- "
                "`pip install xgboost` to reproduce the paper exactly."
            )
            self.model_ = GradientBoostingRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, subsample=self.subsample,
                random_state=self.seed,
            )
        self.model_.fit(X, target)
        return self

    def predict(self, members: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("XGBoostMetaLearner.predict called before fit.")
        n, horizon = members.shape[0], members.shape[1]
        X = np.nan_to_num(self._design(members, context), nan=0.0)
        return _restore(self.model_.predict(X), n, horizon)

    def importances(self) -> pd.DataFrame:
        """Feature importances, sorted -- the interpretability handle of §4.2.2."""
        if self.model_ is None or not hasattr(self.model_, "feature_importances_"):
            return pd.DataFrame(columns=["feature", "importance"])
        return (
            pd.DataFrame(
                {"feature": self.feature_names_, "importance": self.model_.feature_importances_}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# =========================================================================== #
# 4.2.3  Context-Aware Selection
# =========================================================================== #
class HourlyMOS:
    r"""**H-MOS** -- 24 hour-specific linear correctors (paper Eq. 11).

    .. math::  y_{corrected,h} = \alpha_h\, y_{avg,h} + \beta_h

    One ordinary-least-squares fit per hour of day, applied to the members'
    average. PM2.5 has a pronounced diurnal cycle driven by boundary-layer height
    and traffic, and model bias tracks it: a system that is unbiased at noon can
    be 15 % low at the 06:00 inversion peak. A single global calibration cannot
    remove a bias that changes sign through the day; 24 small ones can.

    The cost is 48 parameters, which is why each is fitted on hundreds of samples
    and a fallback to the identity is used when an hour is too sparse -- a
    two-point regression would produce a wild slope and make things worse.
    """

    name = "h_mos"

    def __init__(self, min_samples: int = 10) -> None:
        self.min_samples = min_samples
        self.coefficients_: Dict[int, Tuple[float, float]] = {}

    def fit(
        self,
        members: np.ndarray,
        y: np.ndarray,
        context: Optional[np.ndarray] = None,
        hours: Optional[np.ndarray] = None,
    ):
        if hours is None:
            raise ValueError("HourlyMOS.fit requires the clock hour of every forecast step.")
        average = np.asarray(members, dtype=np.float64).mean(axis=-1).ravel()
        target = np.asarray(y, dtype=np.float64).ravel()
        hour_flat = np.asarray(hours).ravel().astype(int)

        valid = np.isfinite(average) & np.isfinite(target)
        for hour in range(24):
            selector = valid & (hour_flat == hour)
            if selector.sum() < self.min_samples:
                self.coefficients_[hour] = (1.0, 0.0)
                logger.debug(
                    "H-MOS: hour %02d has only %d samples; falling back to the identity.",
                    hour, int(selector.sum()),
                )
                continue
            x, t = average[selector], target[selector]
            if np.std(x) < 1e-9:
                self.coefficients_[hour] = (1.0, float(np.mean(t - x)))
                continue
            slope, intercept = np.polyfit(x, t, 1)
            self.coefficients_[hour] = (float(slope), float(intercept))
        return self

    def predict(
        self,
        members: np.ndarray,
        context: Optional[np.ndarray] = None,
        hours: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self.coefficients_:
            raise RuntimeError("HourlyMOS.predict called before fit.")
        if hours is None:
            raise ValueError("HourlyMOS.predict requires the clock hour of every forecast step.")
        average = np.asarray(members, dtype=np.float64).mean(axis=-1)
        hour_grid = np.asarray(hours).reshape(average.shape).astype(int)

        out = np.empty_like(average)
        for hour in range(24):
            selector = hour_grid == hour
            if not selector.any():
                continue
            slope, intercept = self.coefficients_.get(hour, (1.0, 0.0))
            out[selector] = slope * average[selector] + intercept
        return out

    def table(self) -> pd.DataFrame:
        """The 24 (slope, intercept) pairs -- directly reportable."""
        return pd.DataFrame(
            [{"hour": h, "alpha": a, "beta": b} for h, (a, b) in sorted(self.coefficients_.items())]
        )


class SmartModelSelector:
    r"""**SMS** -- a classifier that picks which member to trust (paper Eqs. 12-13).

    .. math::  i^\ast = C(v_n), \qquad y_{n,SMS} = y_{n,i^\ast}

    A random forest is trained to predict *which member will be most accurate*
    given the meteorological and temporal context, and its choice is used raw --
    no averaging at all.

    This is the strategy with the highest ceiling and the highest risk. Its
    ceiling is the oracle: if the classifier were perfect it would beat every
    blend, because on any given hour the best single member beats their average.
    Its risk is that classification errors cost the *full* gap between members
    rather than being damped by averaging, so a mediocre classifier underperforms
    plain SA. Reporting the classifier's own accuracy alongside the forecast
    metrics is what makes that trade-off visible instead of mysterious.
    """

    name = "sms"

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, seed: int = 42) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.model_ = None
        self.member_names_: List[str] = []
        self.train_accuracy_: float = float("nan")
        self.oracle_gain_: float = float("nan")

    def fit(
        self,
        members: np.ndarray,
        y: np.ndarray,
        context: Optional[np.ndarray] = None,
        member_names: Optional[Sequence[str]] = None,
    ):
        from sklearn.ensemble import RandomForestClassifier

        stacked = _flatten(np.asarray(members, dtype=np.float64))
        target = np.asarray(y, dtype=np.float64).ravel()
        X = (
            np.hstack([np.asarray(context, dtype=np.float64).reshape(stacked.shape[0], -1)])
            if context is not None
            else stacked
        )
        errors = np.abs(stacked - target[:, None])
        mask = np.isfinite(target) & np.isfinite(errors).all(axis=1) & np.isfinite(X).all(axis=1)
        if mask.sum() == 0:
            raise ValueError("SmartModelSelector.fit: no complete rows.")

        labels = np.argmin(errors[mask], axis=1)
        self.member_names_ = (
            list(member_names) if member_names else [f"m{i}" for i in range(stacked.shape[1])]
        )
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            criterion="gini", random_state=self.seed, n_jobs=-1,
        ).fit(X[mask], labels)
        self.train_accuracy_ = float(self.model_.score(X[mask], labels))

        # How much was available to a perfect selector, versus a plain average?
        oracle = errors[mask][np.arange(mask.sum()), labels].mean()
        average = np.abs(stacked[mask].mean(axis=1) - target[mask]).mean()
        self.oracle_gain_ = float(1.0 - oracle / average) if average > 0 else float("nan")
        logger.info(
            "SMS: selector accuracy %.3f on its own training rows; a perfect selector would "
            "cut MAE by %.1f%% versus simple averaging.",
            self.train_accuracy_, self.oracle_gain_ * 100,
        )
        return self

    def predict(self, members: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("SmartModelSelector.predict called before fit.")
        n, horizon = members.shape[0], members.shape[1]
        stacked = _flatten(np.asarray(members, dtype=np.float64))
        X = (
            np.asarray(context, dtype=np.float64).reshape(stacked.shape[0], -1)
            if context is not None
            else stacked
        )
        chosen = self.model_.predict(np.nan_to_num(X, nan=0.0)).astype(int)
        picked = stacked[np.arange(stacked.shape[0]), chosen]
        return _restore(picked, n, horizon)


# =========================================================================== #
def build_ensemble(name: str, cfg, *, seed: int = 42):
    """Instantiate one ensemble from the configuration."""
    ensembles = cfg.ensembles
    if name == "sa":
        return SimpleAveraging()
    if name == "pwe":
        return PerformanceWeightedEnsemble(
            method=ensembles.pwe_method, max_iter=ensembles.pwe_max_iter
        )
    if name == "mlp_meta":
        return MLPMetaLearner(
            hidden=ensembles.mlp_hidden, dropout=ensembles.mlp_dropout,
            learning_rate=ensembles.mlp_learning_rate, epochs=ensembles.mlp_epochs,
            batch_size=ensembles.mlp_batch_size,
            patience=ensembles.mlp_early_stopping_patience, seed=seed,
        )
    if name == "xgb_meta":
        return XGBoostMetaLearner(
            n_estimators=ensembles.xgb_n_estimators, max_depth=ensembles.xgb_max_depth,
            learning_rate=ensembles.xgb_learning_rate, subsample=ensembles.xgb_subsample,
            seed=seed,
        )
    if name == "h_mos":
        return HourlyMOS()
    if name == "sms":
        return SmartModelSelector(
            n_estimators=ensembles.sms_n_estimators, max_depth=ensembles.sms_max_depth, seed=seed
        )
    raise ValueError(f"Unknown ensemble {name!r}; available: {list(AVAILABLE_ENSEMBLES)}")
