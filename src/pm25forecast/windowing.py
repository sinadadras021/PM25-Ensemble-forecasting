"""Segment-aware sliding-window construction.

The naive implementation -- a Python ``for`` loop that re-checks timestamp
continuity inside every candidate window -- is ``O(n * (w + h))`` and dominates
runtime on a decade of hourly data across a dozen stations. The implementation
here is vectorised with :func:`numpy.lib.stride_tricks.sliding_window_view`, so
it is ``O(n)`` in time and allocates only the windows it actually keeps.

Correctness properties, all covered by ``tests/test_windowing.py``:

*   a window never spans two segments;
*   a window never spans a gap, because gaps are what define segments;
*   a window containing any NaN in its **inputs** is rejected, never zero-filled;
*   a window whose **labels** are partially missing is kept when at least
    ``min_label_completeness`` of the horizon is observed, and the missing hours
    stay NaN so that the masked losses and the metrics can ignore them. Requiring
    fully-observed labels sounds stricter but throws away most of the dataset:
    at 92 % hourly coverage, ``0.92 ** 48 ~ 2 %`` of candidates survive;
*   the returned anchor timestamps are the *first input hour* of each window, so
    the closed interval a window occupies is
    ``[anchor, anchor + (input_hours + horizon_hours - 1) hours]``. Split-overlap
    checks depend on that definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .data.segments import SEGMENT_COLUMN
from .logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["WindowSet", "make_windows"]


@dataclass
class WindowSet:
    """Windowed arrays plus the metadata needed to audit and plot them."""

    X: np.ndarray  #: (n, input_hours, n_features) float32
    y: np.ndarray  #: (n, horizon_hours) float32
    anchors: np.ndarray  #: (n,) datetime64[ns] -- first *input* hour
    target_times: np.ndarray  #: (n, horizon_hours) datetime64[ns]
    feature_columns: List[str]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[2]) if self.X.ndim == 3 else 0

    @property
    def first_forecast_times(self) -> np.ndarray:
        """Timestamp of lead time +1 for each window."""
        return self.target_times[:, 0] if len(self) else self.target_times

    def describe(self) -> str:
        if not len(self):
            return "WindowSet(empty)"
        return (
            f"WindowSet(n={len(self)}, steps={self.X.shape[1]}, features={self.n_features}, "
            f"horizon={self.y.shape[1]}, span={pd.Timestamp(self.anchors.min())} -> "
            f"{pd.Timestamp(self.target_times.max())})"
        )


def make_windows(
    scaled: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    target_column: str = "Target",
    input_hours: int = 24,
    horizon_hours: int = 24,
    stride: int = 1,
    segment_column: str = SEGMENT_COLUMN,
    extra_feature_columns: Optional[Sequence[str]] = None,
    min_label_completeness: float = 1.0,
    anchor_hour: Optional[int] = None,
    context: str = "",
) -> WindowSet | Tuple[WindowSet, np.ndarray]:
    """Build ``(X, y)`` windows from a scaled, segmented frame.

    Parameters
    ----------
    scaled:
        Frame produced by :meth:`~pm25forecast.scaling.SplitScaler.transform`, with
        a segment column attached. Must be sorted and hourly.
    feature_columns:
        Ordered encoder inputs. The order is preserved into ``X``.
    stride:
        Hop between consecutive window anchors. Use ``1`` for training (maximum
        sample count) and ``>= horizon_hours`` for test (non-overlapping,
        statistically independent scoring windows).
    extra_feature_columns:
        Optional second feature block returned as a parallel array -- used by the
        meta-learner, which reads a different view of the same windows.
    min_label_completeness:
        Fraction of the horizon that must be observed. ``1.0`` is the strict rule.
    anchor_hour:
        When set, keep only windows whose **first forecast hour** falls on this
        clock hour. With ``anchor_hour=0`` each retained window covers whole
        calendar days, which is what makes the Day 1 / Day 2 daily-mean
        aggregation of paper 4.4 well defined. The stride grid is aligned to this
        hour before it is walked, so a segment that resumes at an arbitrary clock
        hour still contributes every window it can. ``None`` keeps every window,
        which is what base-model *training* uses, where more samples help.

    Returns
    -------
    WindowSet, or ``(WindowSet, extra_array)`` when ``extra_feature_columns`` is
    given.
    """
    missing = [c for c in feature_columns if c not in scaled.columns]
    if missing:
        raise KeyError(f"{context}: feature columns absent from scaled frame: {missing}")
    if target_column not in scaled.columns:
        raise KeyError(f"{context}: target column {target_column!r} absent.")
    if segment_column not in scaled.columns:
        raise KeyError(
            f"{context}: {segment_column!r} absent. Call build_segments before windowing."
        )
    if stride < 1:
        raise ValueError(f"{context}: stride must be >= 1, got {stride}.")
    if not 0.0 < min_label_completeness <= 1.0:
        raise ValueError(
            f"{context}: min_label_completeness must be in (0, 1], got {min_label_completeness}."
        )
    min_valid_labels = int(np.ceil(min_label_completeness * horizon_hours))

    span = input_hours + horizon_hours
    extra_cols = list(extra_feature_columns) if extra_feature_columns else []
    if extra_cols:
        missing_extra = [c for c in extra_cols if c not in scaled.columns]
        if missing_extra:
            raise KeyError(f"{context}: extra feature columns absent: {missing_extra}")

    X_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    extra_blocks: List[np.ndarray] = []
    anchor_blocks: List[np.ndarray] = []
    time_blocks: List[np.ndarray] = []

    considered = 0
    for segment_id, group in scaled.groupby(segment_column, sort=True):
        if segment_id < 0 or len(group) < span:
            continue

        feats = group[list(feature_columns)].to_numpy(dtype=np.float32)
        labels = group[target_column].to_numpy(dtype=np.float32)
        times = group.index.to_numpy()
        extra = group[extra_cols].to_numpy(dtype=np.float32) if extra_cols else None

        # sliding_window_view gives a zero-copy view; slicing with the stride and
        # only then calling np.ascontiguousarray keeps peak memory proportional to
        # the number of *kept* windows rather than to n * window_size.
        feat_windows = sliding_window_view(feats, input_hours, axis=0)  # (m, F, W)
        label_windows = sliding_window_view(labels, horizon_hours, axis=0)  # (k, H)
        time_windows = sliding_window_view(times, horizon_hours, axis=0)

        n_candidates = len(group) - span + 1

        # The stride grid must be *aligned* to the anchor hour before it is walked,
        # not filtered afterwards. A segment begins at whatever clock hour the
        # record happens to resume at, so an unaligned grid 0, stride, 2*stride, ...
        # lands on one fixed clock hour per segment. When that hour is not the
        # anchor hour the whole segment is discarded, and a record broken into many
        # short segments loses almost every window -- silently, because each segment
        # merely contributes nothing. Offsetting the grid to the first admissible
        # anchor keeps the daily cadence and the anchor rule from cancelling out.
        offset = 0
        if anchor_hour is not None and n_candidates > 0:
            first_forecast_hours = pd.DatetimeIndex(
                pd.to_datetime(times[input_hours : input_hours + n_candidates])
            ).hour.to_numpy()
            admissible = np.flatnonzero(first_forecast_hours == anchor_hour)
            if admissible.size == 0:
                continue
            offset = int(admissible[0])

        starts = np.arange(offset, n_candidates, stride)
        if starts.size == 0:
            continue
        considered += starts.size

        Xw = feat_windows[starts].transpose(0, 2, 1)  # (n, W, F)
        yw = label_windows[starts + input_hours]  # (n, H)
        tw = time_windows[starts + input_hours]  # (n, H)

        # Inputs must be complete; labels only need to be complete *enough*.
        valid = ~np.isnan(Xw).any(axis=(1, 2))
        valid &= (~np.isnan(yw)).sum(axis=1) >= min_valid_labels
        if anchor_hour is not None:
            first_forecast = pd.DatetimeIndex(pd.to_datetime(tw[:, 0]))
            valid &= np.asarray(first_forecast.hour == anchor_hour, dtype=bool)
        if extra is not None:
            extra_windows = sliding_window_view(extra, input_hours, axis=0)
            Ew = extra_windows[starts].transpose(0, 2, 1)
            valid &= ~np.isnan(Ew).any(axis=(1, 2))

        if not valid.any():
            continue

        X_blocks.append(np.ascontiguousarray(Xw[valid]))
        y_blocks.append(np.ascontiguousarray(yw[valid]))
        anchor_blocks.append(times[starts[valid]])
        time_blocks.append(np.ascontiguousarray(tw[valid]))
        if extra is not None:
            extra_blocks.append(np.ascontiguousarray(Ew[valid]))

    if X_blocks:
        window_set = WindowSet(
            X=np.concatenate(X_blocks, axis=0),
            y=np.concatenate(y_blocks, axis=0),
            anchors=np.concatenate(anchor_blocks, axis=0),
            target_times=np.concatenate(time_blocks, axis=0),
            feature_columns=list(feature_columns),
        )
    else:
        window_set = WindowSet(
            X=np.empty((0, input_hours, len(feature_columns)), dtype=np.float32),
            y=np.empty((0, horizon_hours), dtype=np.float32),
            anchors=np.array([], dtype="datetime64[ns]"),
            target_times=np.empty((0, horizon_hours), dtype="datetime64[ns]"),
            feature_columns=list(feature_columns),
        )

    kept = len(window_set)
    label_coverage = (
        float(np.mean(~np.isnan(window_set.y))) if kept else float("nan")
    )
    logger.info(
        "%s: %d/%d candidate windows kept (%.1f%%) at stride %d%s; label coverage %.1f%%.",
        context, kept, considered, kept / max(considered, 1) * 100, stride,
        f", anchored at {anchor_hour:02d}:00" if anchor_hour is not None else "",
        label_coverage * 100,
    )
    if considered and kept == 0:
        logger.warning(
            "%s: no candidate window survived. Either an input feature has a NaN in every "
            "window (check imputation coverage and whether a column is mostly empty), or "
            "fewer than %d of the %d horizon hours are ever observed together.",
            context, min_valid_labels, horizon_hours,
        )

    if extra_cols:
        extra_array = (
            np.concatenate(extra_blocks, axis=0)
            if extra_blocks
            else np.empty((0, input_hours, len(extra_cols)), dtype=np.float32)
        )
        return window_set, extra_array
    return window_set
