r"""Evaluation metrics (paper §4.4, Eqs. 14-18).

The hourly forecasts are aggregated into **daily means** and reported separately
for **Day 1** (lead 1-24 h) and **Day 2** (lead 25-48 h). That structure is the
point, not a formatting choice: a single 48-hour average hides the decay of
predictability with lead time, and it is precisely the Day-2 number that decides
whether a warning system is useful for planning rather than only for reacting.

Five statistics, exactly as defined in the paper:

===========  ==========================================================
metric       Eq.  meaning
===========  ==========================================================
``RMSE``     14   error magnitude, penalising large misses quadratically
``MAE``      15   error magnitude, linear
``R``        16   **Pearson correlation** -- shape agreement, blind to bias
``IOA``      17   Willmott index of agreement -- bounded, bias-sensitive
``Unc%``     18   uncertainty: spread of the error about the MAE
===========  ==========================================================

``R`` and ``IOA`` are reported together on purpose. A forecast that tracks every
rise and fall but sits 20 ug/m3 too high has ``R`` near 1 and a poor ``IOA``;
seeing both makes the difference between "wrong shape" and "right shape, wrong
level" immediately legible.

Every function masks NaNs rather than imputing them, and returns ``nan`` instead
of raising when a metric is undefined on the given sample.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "continuous_metrics",
    "event_metrics",
    "daily_means",
    "split_days",
    "summarise_forecast",
    "metrics_table",
]


def _paired(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: y_true {a.shape} vs y_pred {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def continuous_metrics(y_true, y_pred, *, prefix: str = "") -> Dict[str, float]:
    r"""RMSE, MAE, R, IOA and Unc% (paper Eqs. 14-18), plus bias and R^2.

    ``MBE`` and ``R2`` are not in the paper's list but cost nothing and answer the
    first two questions a reader forms on seeing an RMSE: is the error one-sided,
    and does the forecast explain more variance than the observed mean would.
    """
    a, b = _paired(y_true, y_pred)
    p = prefix
    keys = ("RMSE", "MAE", "R", "IOA", "Unc_percent", "MBE", "R2")
    if a.size == 0:
        return {f"{p}n": 0, **{f"{p}{k}": float("nan") for k in keys}}

    error = b - a
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    obs_mean = float(np.mean(a))

    # Eq. 16 -- Pearson correlation.
    if a.size > 1 and np.std(a) > 0 and np.std(b) > 0:
        r = float(np.corrcoef(a, b)[0, 1])
    else:
        r = float("nan")

    # Eq. 17 -- Willmott's index of agreement.
    denominator = float(np.sum((np.abs(b - obs_mean) + np.abs(a - obs_mean)) ** 2))
    ss_res = float(np.sum(error**2))
    ioa = 1.0 - ss_res / denominator if denominator > 0 else float("nan")

    # Eq. 18 -- uncertainty: RMS spread of |error| about the MAE, relative to the
    # observed mean. Small Unc% means the errors are consistent in size; large
    # Unc% means occasional very large misses hide behind an acceptable MAE.
    unc = (
        float(np.sqrt(np.mean((np.abs(error) - mae) ** 2)) / obs_mean * 100.0)
        if obs_mean > 0
        else float("nan")
    )

    ss_tot = float(np.sum((a - obs_mean) ** 2))
    return {
        f"{p}n": int(a.size),
        f"{p}RMSE": rmse,
        f"{p}MAE": mae,
        f"{p}R": r,
        f"{p}IOA": ioa,
        f"{p}Unc_percent": unc,
        f"{p}MBE": float(np.mean(error)),
        f"{p}R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def event_metrics(y_true, y_pred, *, threshold: float, prefix: str = "") -> Dict[str, float]:
    """Contingency scores for exceedance of ``threshold`` (**beyond the paper**).

    Reported because a daily-mean RMSE says nothing about whether the days that
    breached the guideline were the days that got flagged. ``CSI`` ignores true
    negatives and is the honest headline when exceedances are a minority.
    """
    a, b = _paired(y_true, y_pred)
    p = prefix
    keys = ("TP", "FP", "FN", "TN", "POD", "FAR", "precision", "F1", "CSI", "base_rate")
    if a.size == 0:
        return {f"{p}{k}": float("nan") for k in keys}

    obs, pred = a >= threshold, b >= threshold
    tp = int(np.sum(obs & pred))
    fp = int(np.sum(~obs & pred))
    fn = int(np.sum(obs & ~pred))
    tn = int(np.sum(~obs & ~pred))
    total = tp + fp + fn + tn

    pod = tp / (tp + fn) if (tp + fn) else float("nan")
    far = fp / (tp + fp) if (tp + fp) else float("nan")
    precision = 1.0 - far if np.isfinite(far) else float("nan")
    f1 = (
        2 * precision * pod / (precision + pod)
        if np.isfinite(precision) and np.isfinite(pod) and (precision + pod) > 0
        else float("nan")
    )
    return {
        f"{p}TP": tp, f"{p}FP": fp, f"{p}FN": fn, f"{p}TN": tn,
        f"{p}POD": pod, f"{p}FAR": far, f"{p}precision": precision,
        f"{p}F1": f1,
        f"{p}CSI": tp / (tp + fp + fn) if (tp + fp + fn) else float("nan"),
        f"{p}base_rate": (tp + fn) / total if total else float("nan"),
    }


# --------------------------------------------------------------------------- #
def daily_means(series_2d: np.ndarray, *, min_hours: int = 18) -> np.ndarray:
    """Mean of each consecutive 24-hour block, with a completeness requirement.

    Parameters
    ----------
    series_2d:
        ``(n_windows, horizon)`` hourly values; the horizon must be a whole
        number of days.

    Returns
    -------
    ``(n_windows, horizon // 24)`` daily means; a day with fewer than
    ``min_hours`` observed values is NaN rather than a silently short average.

    >>> daily_means(np.ones((1, 48))).tolist()
    [[1.0, 1.0]]
    """
    array = np.asarray(series_2d, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"daily_means expects 2-D input, got {array.shape}")
    horizon = array.shape[1]
    if horizon % 24:
        raise ValueError(f"horizon ({horizon}) is not a whole number of days.")

    n_days = horizon // 24
    out = np.empty((array.shape[0], n_days), dtype=np.float64)
    for day in range(n_days):
        block = array[:, day * 24 : (day + 1) * 24]
        valid = np.isfinite(block).sum(axis=1)
        with np.errstate(invalid="ignore"):
            means = np.nanmean(block, axis=1)
        out[:, day] = np.where(valid >= min_hours, means, np.nan)
    return out


def split_days(series_2d: np.ndarray, *, min_hours: int = 18) -> Dict[str, np.ndarray]:
    """Label each daily mean ``day1``, ``day2``, ... (paper §4.4)."""
    means = daily_means(series_2d, min_hours=min_hours)
    return {f"day{index + 1}": means[:, index] for index in range(means.shape[1])}


def summarise_forecast(
    y_true_2d: np.ndarray,
    y_pred_2d: np.ndarray,
    *,
    min_hours_per_day: int = 18,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """One results row: hourly metrics plus per-day daily-mean metrics.

    The paper reports the daily-mean numbers; the hourly block is included
    because a reader will want to know whether a good daily mean was achieved by
    tracking the diurnal cycle or by averaging two opposite errors.
    """
    out: Dict[str, float] = {}
    out.update(continuous_metrics(y_true_2d, y_pred_2d, prefix="hourly_"))

    truth_days = split_days(y_true_2d, min_hours=min_hours_per_day)
    pred_days = split_days(y_pred_2d, min_hours=min_hours_per_day)

    for label in truth_days:
        out.update(continuous_metrics(truth_days[label], pred_days[label], prefix=f"{label}_"))
        if threshold is not None:
            out.update(
                event_metrics(
                    truth_days[label], pred_days[label],
                    threshold=threshold, prefix=f"{label}_",
                )
            )
    return out


def metrics_table(rows: Iterable[Mapping[str, object]], *, sort_by: Optional[str] = None):
    """Assemble result dicts into a tidy frame with stable column ordering."""
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return frame
    leading = [c for c in ("station", "slug", "strategy", "model") if c in frame.columns]
    rest = [c for c in frame.columns if c not in leading]
    frame = frame[leading + rest]
    if sort_by and sort_by in frame.columns:
        frame = frame.sort_values(sort_by, kind="mergesort")
    return frame.reset_index(drop=True)
