"""Windowing rules that the reported sample size depends on.

A window is the unit of scoring. If windows are built wrongly the metrics are
still computed, still look reasonable, and are still wrong -- so these rules are
asserted rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pm25forecast.data.segments import SEGMENT_COLUMN
from pm25forecast.windowing import make_windows

INPUT_HOURS = 48
HORIZON_HOURS = 48
SPAN = INPUT_HOURS + HORIZON_HOURS


def _frame(start: str, hours: int, segment_id: int = 0) -> pd.DataFrame:
    index = pd.date_range(start, periods=hours, freq="h")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "feat_a": rng.normal(size=hours),
            "feat_b": rng.normal(size=hours),
            "Target": rng.normal(size=hours),
        },
        index=index,
    )
    frame[SEGMENT_COLUMN] = segment_id
    return frame


def _windows(frame, **kwargs):
    return make_windows(
        frame,
        ["feat_a", "feat_b"],
        input_hours=INPUT_HOURS,
        horizon_hours=HORIZON_HOURS,
        context="test",
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# The anchor rule and the stride grid must not cancel each other out
# --------------------------------------------------------------------------- #
def test_anchored_windows_survive_a_segment_that_starts_off_the_anchor_hour():
    """A segment beginning at an arbitrary clock hour must still yield windows.

    Real records resume at whatever hour the instrument came back. If the stride
    grid is walked from the segment's first row without being aligned to the
    anchor hour first, every anchor in that segment lands on one fixed clock
    hour -- and when that hour is not the anchor hour the entire segment is
    discarded. The failure is silent: each segment merely contributes nothing,
    so a shattered record loses almost all of its windows while every log line
    still reports success.
    """
    # Starts at 07:00, i.e. deliberately off the midnight anchor.
    frame = _frame("2021-03-04 07:00", hours=SPAN * 6)

    result = _windows(frame, stride=HORIZON_HOURS, anchor_hour=0)
    window_set = result[0] if isinstance(result, tuple) else result

    assert len(window_set) > 0, (
        "a segment that does not happen to begin at the anchor hour produced no "
        "windows -- the stride grid is not aligned to the anchor"
    )
    first_forecast = pd.DatetimeIndex(pd.to_datetime(window_set.target_times[:, 0]))
    assert (first_forecast.hour == 0).all()


def test_anchored_yield_matches_the_number_of_whole_days_available():
    """Alignment must not merely produce *some* windows -- it must produce all."""
    days = 8
    frame = _frame("2021-03-04 07:00", hours=24 * days + SPAN)

    result = _windows(frame, stride=HORIZON_HOURS, anchor_hour=0)
    window_set = result[0] if isinstance(result, tuple) else result

    # The grid starts at the first midnight-anchored candidate and then advances
    # by the stride, so the expected count is every stride-th admissible anchor --
    # not every admissible anchor, which would overlap when stride > 24.
    index = frame.index
    n_candidates = len(index) - SPAN + 1
    offset = next(
        i for i in range(n_candidates) if index[i + INPUT_HOURS].hour == 0
    )
    expected = len(range(offset, n_candidates, HORIZON_HOURS))
    assert len(window_set) == expected
    assert expected > 1, "the fixture must exercise more than a single window"


def test_scored_windows_never_share_a_labelled_hour():
    """Non-overlapping labels are what make the reported sample size honest.

    With stride = horizon the label blocks tile the record instead of repeating
    it, so an observation is scored once. A smaller stride would count correlated
    repeats as independent samples and narrow every interval derived from them.
    """
    frame = _frame("2021-03-04 00:00", hours=SPAN * 8)

    result = _windows(frame, stride=HORIZON_HOURS, anchor_hour=0)
    window_set = result[0] if isinstance(result, tuple) else result

    labelled = window_set.target_times.reshape(-1)
    assert len(labelled) == len(set(labelled.tolist())), "a labelled hour is scored twice"


def test_a_window_never_spans_two_segments():
    """Segments exist because the hours across a break are not adjacent."""
    left = _frame("2021-03-04 00:00", hours=SPAN + 12, segment_id=0)
    right = _frame("2021-06-01 00:00", hours=SPAN + 12, segment_id=1)
    frame = pd.concat([left, right])

    result = _windows(frame, stride=1)
    window_set = result[0] if isinstance(result, tuple) else result

    for row in window_set.target_times:
        span_hours = (pd.Timestamp(row[-1]) - pd.Timestamp(row[0])).total_seconds() / 3600
        assert span_hours == HORIZON_HOURS - 1


def test_a_window_with_a_missing_input_is_rejected_not_zero_filled():
    """A zero-filled input is an invented observation the model cannot detect."""
    frame = _frame("2021-03-04 00:00", hours=SPAN * 4)
    frame.iloc[10, frame.columns.get_loc("feat_a")] = np.nan

    result = _windows(frame, stride=1)
    window_set = result[0] if isinstance(result, tuple) else result

    assert not np.isnan(window_set.X).any()
    assert len(window_set) < len(frame) - SPAN + 1
