"""Contiguity segmentation.

A sliding window is only meaningful if the hours inside it are actually
consecutive *and observed*. After a 40-hour outage, hour 41 sits next to hour 0
in the array but three days apart in the atmosphere. Feeding that to a recurrent
model teaches it a discontinuity that never happens in reality.

Two distinct things can break contiguity, and both are handled here:

* **a NaN** in a column the window needs, and
* **a hole in the index itself**, which is what row-dropping produces.

The second is the subtle one. Once unusable rows have been deleted, the frame
contains no NaN at all, so a NaN-only check reports one long clean segment while
the timestamps actually jump across the deleted hours. A window built there
splices two different days together and nothing downstream notices. Comparing
consecutive timestamps catches it.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["SEGMENT_COLUMN", "build_segments", "segment_summary"]

SEGMENT_COLUMN = "Segment_ID"


def build_segments(
    df: pd.DataFrame,
    critical_columns: Sequence[str],
    *,
    max_gap_hours: int = 12,
    freq_hours: int = 1,
    context: str = "",
) -> pd.DataFrame:
    """Label maximal runs of usable rows, split by long gaps in critical columns.

    A row is *unusable* when any critical column is NaN. Consecutive unusable
    rows form a gap; a gap **longer than** ``max_gap_hours`` ends the current
    segment. Shorter gaps are tolerated because the imputation stage is expected
    to fill them, and the window builder drops any window that still contains a
    NaN.

    Independently, a jump of more than ``max_gap_hours`` between consecutive
    *timestamps* also ends the segment. This is what protects a frame whose
    unusable rows have already been deleted: such a frame has no NaN left, so a
    NaN-only rule would happily build windows straight across the deleted hours.

    The returned frame gains one integer column, :data:`SEGMENT_COLUMN`. Rows
    inside a long gap are given ``-1``: they belong to no segment and are never
    used as window material.

    Notes
    -----
    Segment identifiers are dense and increasing (``0, 1, 2, ...``) so they can
    be used directly as group keys and appear predictably in diagnostics. This is
    a deliberate departure from the more common ``cumsum`` trick, which produces
    one spurious identifier per gap hour and makes segment counts meaningless.

    Examples
    --------
    >>> idx = pd.date_range("2024-01-01", periods=8, freq="h")
    >>> df = pd.DataFrame({"y": [1, 2, np.nan, np.nan, np.nan, 3, 4, 5]}, index=idx)
    >>> build_segments(df, ["y"], max_gap_hours=2)["Segment_ID"].tolist()
    [0, 0, -1, -1, -1, 1, 1, 1]
    """
    if df.empty:
        out = df.copy()
        out[SEGMENT_COLUMN] = np.array([], dtype=np.int64)
        return out

    present = [c for c in critical_columns if c in df.columns]
    if not present:
        raise KeyError(
            f"{context}: none of the critical columns {list(critical_columns)} exist in the frame."
        )

    unusable = df[present].isna().any(axis=1).to_numpy()

    # A jump in the index is a break in its own right, regardless of NaN.
    step = pd.Timedelta(hours=freq_hours)
    deltas = pd.Series(df.index).diff().to_numpy()
    index_break = np.zeros(len(df), dtype=bool)
    if len(df) > 1:
        index_break[1:] = deltas[1:] > (step * (max_gap_hours + 1))
    n_index_breaks = int(index_break.sum())
    if n_index_breaks:
        logger.info(
            "%s: %d discontinuities in the timestamp index will also split segments.",
            context, n_index_breaks,
        )

    # Identify runs of identical usability, then measure the length of each
    # unusable run to decide whether it breaks the series.
    run_id = np.concatenate([[0], np.cumsum(unusable[1:] != unusable[:-1])])
    run_lengths = pd.Series(np.ones(len(unusable))).groupby(run_id).transform("sum").to_numpy()

    breaking_gap = unusable & (run_lengths > max_gap_hours)

    segment = np.full(len(df), -1, dtype=np.int64)
    current = -1
    open_segment = False
    for i in range(len(df)):
        if breaking_gap[i]:
            open_segment = False
            continue
        if index_break[i]:
            open_segment = False
        if not open_segment:
            current += 1
            open_segment = True
        segment[i] = current

    out = df.copy()
    out[SEGMENT_COLUMN] = segment

    n_segments = current + 1
    usable = int((segment >= 0).sum())
    logger.info(
        "%s: %d segments covering %d/%d rows (%.1f%%); longest gap policy = %dh",
        context, n_segments, usable, len(df),
        usable / max(len(df), 1) * 100, max_gap_hours,
    )
    return out


def segment_summary(df: pd.DataFrame, *, min_length: int = 0) -> pd.DataFrame:
    """Per-segment length table, useful for diagnosing 'zero windows' failures."""
    if SEGMENT_COLUMN not in df.columns:
        raise KeyError(f"{SEGMENT_COLUMN!r} not present; call build_segments first.")
    valid = df[df[SEGMENT_COLUMN] >= 0]
    if valid.empty:
        return pd.DataFrame(columns=["segment", "start", "end", "length_hours"])
    grouped = valid.groupby(SEGMENT_COLUMN)
    table = pd.DataFrame(
        {
            "segment": grouped.size().index,
            "start": grouped.apply(lambda g: g.index.min()).to_numpy(),
            "end": grouped.apply(lambda g: g.index.max()).to_numpy(),
            "length_hours": grouped.size().to_numpy(),
        }
    ).reset_index(drop=True)
    return table[table["length_hours"] >= min_length].sort_values(
        "length_hours", ascending=False
    ).reset_index(drop=True)
