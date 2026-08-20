"""Reference forecasts (**beyond the paper**).

The paper compares its six ensembles against the four base learners and does not
report naive baselines, so ``evaluation.baselines`` is empty by default. These
are available as an opt-in because a reviewer will eventually ask whether the
network is adding information or re-learning the diurnal cycle.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["AVAILABLE_BASELINES", "HourClimatology", "persistence_forecast", "build_baselines"]

AVAILABLE_BASELINES = ("persistence", "climatology", "persistence_anomaly")


class HourClimatology:
    """Mean target per (month, hour), fitted on the training block only."""

    def __init__(self) -> None:
        self._cell: Dict[Tuple[int, int], float] = {}
        self._by_month: Dict[int, float] = {}
        self._by_hour: Dict[int, float] = {}
        self._global = float("nan")

    def fit(self, series: pd.Series) -> "HourClimatology":
        observed = series.dropna()
        if observed.empty:
            raise ValueError("HourClimatology.fit: no observed values.")
        index = pd.DatetimeIndex(observed.index)
        frame = pd.DataFrame(
            {"value": observed.to_numpy(), "month": index.month, "hour": index.hour}
        )
        self._cell = frame.groupby(["month", "hour"])["value"].mean().to_dict()
        self._by_month = frame.groupby("month")["value"].mean().to_dict()
        self._by_hour = frame.groupby("hour")["value"].mean().to_dict()
        self._global = float(frame["value"].mean())
        return self

    def predict(self, times) -> np.ndarray:
        shape = np.asarray(times).shape
        index = pd.DatetimeIndex(pd.to_datetime(np.asarray(times).ravel()))
        out = np.empty(len(index), dtype=float)
        for i, (month, hour) in enumerate(zip(index.month, index.hour)):
            value = self._cell.get((month, hour))
            if value is None or not np.isfinite(value):
                value = self._by_month.get(month, self._by_hour.get(hour, self._global))
            out[i] = value
        return out.reshape(shape)


def persistence_forecast(history_2d: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last ``horizon`` observed hours.

    With a 48-hour window and a 48-hour horizon this is "the same hour two days
    ago", a genuinely competitive reference for a pollutant with a strong,
    repeating diurnal cycle.
    """
    history = np.asarray(history_2d, dtype=float)
    if history.ndim != 2:
        raise ValueError(f"persistence_forecast expects 2-D history, got {history.shape}")
    if history.shape[1] >= horizon:
        return history[:, -horizon:].copy()
    reps = int(np.ceil(horizon / history.shape[1]))
    return np.tile(history, (1, reps))[:, -horizon:].copy()


def build_baselines(
    which: Sequence[str],
    *,
    train_target: pd.Series,
    history: np.ndarray,
    input_times: np.ndarray,
    target_times: np.ndarray,
    horizon: int,
    clip: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    """Compute every requested baseline for one windowed test set."""
    unknown = sorted(set(which) - set(AVAILABLE_BASELINES))
    if unknown:
        raise ValueError(f"Unknown baselines {unknown}; available: {list(AVAILABLE_BASELINES)}")
    if not which:
        return {}

    out: Dict[str, np.ndarray] = {}
    climatology: Optional[HourClimatology] = None
    if {"climatology", "persistence_anomaly"} & set(which):
        climatology = HourClimatology().fit(train_target)

    for name in which:
        if name == "persistence":
            out[name] = persistence_forecast(history, horizon)
        elif name == "climatology":
            out[name] = climatology.predict(target_times)
        elif name == "persistence_anomaly":
            with np.errstate(invalid="ignore"):
                anomaly = np.nanmean(
                    np.asarray(history, float) - climatology.predict(input_times),
                    axis=1, keepdims=True,
                )
            out[name] = climatology.predict(target_times) + np.nan_to_num(anomaly, nan=0.0)

    if clip is not None:
        for name in out:
            out[name] = np.clip(out[name], clip[0], clip[1])
    logger.info("Computed %d optional baseline forecasts: %s", len(out), ", ".join(out))
    return out
