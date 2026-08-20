"""Runtime integrity guards.

Every function here answers one question: *is this array/frame still what the
science says it should be?* They are cheap, they run inside the pipeline (not
only in the test suite), and they fail loudly.

The philosophy is that a silent methodological error is far more expensive than
a crash. A crash costs an afternoon; a leaked test set costs a retraction.

All guards raise :class:`IntegrityError`, which subclasses ``AssertionError`` so
that ``pytest.raises(AssertionError)`` in downstream projects keeps working.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "IntegrityError",
    "check_monotonic_hourly_index",
    "check_no_duplicate_timestamps",
    "check_columns_present",
    "check_splits_disjoint",
    "check_no_split_overlap_after_windowing",
    "check_scaler_fitted_on_train_only",
    "check_finite",
    "check_window_shapes",
    "check_target_not_imputed",
    "summarise_missing",
]


class IntegrityError(AssertionError):
    """A pipeline invariant was violated."""


# --------------------------------------------------------------------------- #
# Index / frame guards
# --------------------------------------------------------------------------- #
def check_no_duplicate_timestamps(df: pd.DataFrame, *, context: str = "") -> None:
    """Duplicated timestamps silently corrupt ``reindex`` and every rolling stat."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise IntegrityError(f"{context}: index must be a DatetimeIndex, got {type(df.index)!r}")
    duplicated = df.index.duplicated()
    if duplicated.any():
        examples = df.index[duplicated][:5].tolist()
        raise IntegrityError(
            f"{context}: {int(duplicated.sum())} duplicated timestamps "
            f"(first few: {examples}). Deduplicate before reindexing."
        )


def check_monotonic_hourly_index(
    df: pd.DataFrame, *, context: str = "", freq: str = "h", strict: bool = True
) -> None:
    """Assert a gapless, strictly increasing index at the given frequency."""
    check_no_duplicate_timestamps(df, context=context)
    if not df.index.is_monotonic_increasing:
        raise IntegrityError(f"{context}: index is not monotonically increasing.")
    if len(df) < 2:
        return
    expected = pd.date_range(df.index[0], df.index[-1], freq=freq)
    if strict and len(expected) != len(df):
        missing = expected.difference(df.index)
        raise IntegrityError(
            f"{context}: index has {len(df)} rows but the span {df.index[0]} -> {df.index[-1]} "
            f"contains {len(expected)} {freq} steps. {len(missing)} timestamps are missing; "
            f"reindex onto a complete grid first (otherwise 'limit=N' in interpolate() counts "
            f"rows, not hours)."
        )


def check_columns_present(
    df: pd.DataFrame, required: Iterable[str], *, context: str = ""
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise IntegrityError(
            f"{context}: required columns absent: {missing}. Present: {list(df.columns)}"
        )


def summarise_missing(df: pd.DataFrame, *, top: int = 10) -> pd.DataFrame:
    """Per-column missingness, sorted worst-first. Logged, never raised."""
    frac = df.isna().mean().sort_values(ascending=False)
    out = frac.head(top).to_frame("missing_fraction")
    out["missing_count"] = df.isna().sum().reindex(out.index)
    return out


# --------------------------------------------------------------------------- #
# Split / leakage guards
# --------------------------------------------------------------------------- #
def check_splits_disjoint(
    splits: Mapping[str, Tuple[pd.Timestamp, pd.Timestamp]], *, context: str = ""
) -> None:
    """No two chronological blocks may overlap, in any order."""
    items = sorted(splits.items(), key=lambda kv: kv[1][0])
    for (name_a, (_, end_a)), (name_b, (start_b, _)) in zip(items, items[1:]):
        if start_b <= end_a:
            raise IntegrityError(
                f"{context}: splits {name_a!r} and {name_b!r} overlap "
                f"({name_a} ends {end_a}, {name_b} starts {start_b})."
            )


def check_no_split_overlap_after_windowing(
    window_starts: Mapping[str, np.ndarray],
    *,
    input_hours: int,
    horizon_hours: int,
    purge_hours: int = 0,
    context: str = "",
) -> None:
    """The subtle one.

    A window anchored inside ``train`` spans ``input_hours + horizon_hours``
    into the future. If ``train`` ends the hour before ``val`` begins, the last
    few training windows physically read validation timestamps. This guard
    reconstructs each window's *closed time interval* and asserts that the
    intervals of different splits stay ``purge_hours`` apart.

    Parameters
    ----------
    window_starts:
        Mapping split name -> array of ``datetime64`` anchors (the timestamp of
        the first input step of each window).
    """
    span = pd.Timedelta(hours=input_hours + horizon_hours - 1)
    purge = pd.Timedelta(hours=purge_hours)

    intervals = {}
    for name, starts in window_starts.items():
        if starts is None or len(starts) == 0:
            continue
        starts = pd.DatetimeIndex(pd.to_datetime(starts))
        intervals[name] = (starts.min(), starts.max() + span)

    ordered = sorted(intervals.items(), key=lambda kv: kv[1][0])
    for (name_a, (_, end_a)), (name_b, (start_b, _)) in zip(ordered, ordered[1:]):
        if start_b - end_a < purge:
            raise IntegrityError(
                f"{context}: windows from split {name_a!r} reach {end_a} while split "
                f"{name_b!r} begins at {start_b}; the required purge is {purge_hours}h but the "
                f"actual separation is {(start_b - end_a).total_seconds() / 3600:.1f}h. "
                f"Increase split.purge_hours or shrink the split boundary."
            )


def check_scaler_fitted_on_train_only(
    scaler,
    fit_index: pd.DatetimeIndex,
    train_span: Tuple[pd.Timestamp, pd.Timestamp],
    *,
    context: str = "",
) -> None:
    """Assert every row used to fit a scaler lies inside the training block."""
    if len(fit_index) == 0:
        raise IntegrityError(f"{context}: scaler was fitted on zero rows.")
    start, end = train_span
    outside = fit_index[(fit_index < start) | (fit_index > end)]
    if len(outside):
        raise IntegrityError(
            f"{context}: scaler {type(scaler).__name__} was fitted on {len(outside)} rows "
            f"outside the training block [{start}, {end}] "
            f"(e.g. {outside[:3].tolist()}). This is target leakage."
        )


def check_target_not_imputed(
    original: pd.Series, processed: pd.Series, *, allow: bool, context: str = ""
) -> None:
    """Guard the ground truth.

    Filling gaps in the *target* with a model and then scoring against those
    filled values measures the imputer, not the forecaster. This raises unless
    the caller explicitly opted in.
    """
    fabricated = original.isna() & processed.notna()
    n = int(fabricated.sum())
    if n == 0:
        return
    message = (
        f"{context}: imputation created {n} target values that were never observed "
        f"({n / max(len(original), 1) * 100:.2f}% of rows)."
    )
    if allow:
        logger.warning("%s Proceeding because imputation.impute_target=True.", message)
        return
    raise IntegrityError(
        message + " Set imputation.impute_target=True only if you intend to score against "
        "synthetic ground truth, and say so in the paper."
    )


# --------------------------------------------------------------------------- #
# Array guards
# --------------------------------------------------------------------------- #
def check_finite(array: np.ndarray, *, name: str = "array", allow_nan: bool = False) -> None:
    arr = np.asarray(array)
    if not allow_nan and np.isnan(arr).any():
        idx = np.argwhere(np.isnan(arr))[:3].tolist()
        raise IntegrityError(f"{name}: contains NaN (first positions {idx}).")
    if np.isinf(arr).any():
        idx = np.argwhere(np.isinf(arr))[:3].tolist()
        raise IntegrityError(f"{name}: contains +/-inf (first positions {idx}).")


def check_window_shapes(
    X: np.ndarray,
    y: np.ndarray,
    *,
    input_hours: int,
    horizon_hours: int,
    n_features: Optional[int] = None,
    context: str = "",
) -> None:
    """Shape contract for every windowed dataset."""
    if X.ndim != 3:
        raise IntegrityError(f"{context}: X must be 3-D (samples, steps, features); got {X.shape}")
    if y.ndim not in (2, 3):
        raise IntegrityError(f"{context}: y must be 2-D or 3-D; got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise IntegrityError(
            f"{context}: sample-count mismatch X={X.shape[0]} vs y={y.shape[0]}"
        )
    if X.shape[1] != input_hours:
        raise IntegrityError(
            f"{context}: X has {X.shape[1]} input steps, config says {input_hours}"
        )
    if y.shape[1] != horizon_hours:
        raise IntegrityError(
            f"{context}: y has {y.shape[1]} horizon steps, config says {horizon_hours}"
        )
    if n_features is not None and X.shape[2] != n_features:
        raise IntegrityError(
            f"{context}: X has {X.shape[2]} features, expected {n_features}"
        )
    if X.shape[0] == 0:
        raise IntegrityError(
            f"{context}: zero windows were produced. Usual causes: the split is shorter than "
            f"input+horizon, every candidate window contained a NaN in its inputs, "
            f"windows.min_label_completeness is set too high for this station's coverage, or "
            f"imputation.segment_break_hours is too small."
        )
    if np.isnan(X).any():
        raise IntegrityError(
            f"{context}: encoder inputs contain NaN. Inputs must be complete; only labels may "
            f"be partially missing."
        )
    observed = float(np.mean(~np.isnan(y)))
    if observed == 0.0:
        raise IntegrityError(f"{context}: no label in any window is observed.")
