r"""Grid regularisation and IQR outlier removal (paper §2.2).

The outlier rule is the interquartile range with :math:`k = 1.5`: values outside
:math:`[Q_1 - k\,\mathrm{IQR},\; Q_3 + k\,\mathrm{IQR}]` are **excluded** from the
dataset.

Two implementation properties are enforced here that a direct reading of the
method easily loses, and both change the reported numbers materially:

**Bounds come from the training block only.** Computing quantiles over the whole
record lets the test year shape its own preprocessing. Computing them separately
*within* the test year is worse still: the train and test targets then undergo
different transformations, so the model is scored against a differently-truncated
ground truth than the one it was fitted to.

**Excluded values become NaN and are then imputed; they are never clipped to the
boundary.** Clipping converts a detectable fault into a plausible-looking
observation. Applied to the *target*, it truncates exactly the pollution episodes
the model exists to predict, and the resulting metrics flatter the model on the
days that matter most.

**Validation and test labels are never screened at all**, whatever the
configuration says. A ground truth that has been altered is no longer a ground
truth, and this is enforced in code rather than left to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..validation import IntegrityError, check_monotonic_hourly_index

logger = get_logger(__name__)

__all__ = [
    "QualityReport",
    "reindex_to_hourly",
    "IQRScreen",
    "assess_quality",
    "drop_unusable_columns",
]


@dataclass
class QualityReport:
    """Why a station was admitted or not, ready for the supplementary material."""

    station: str
    n_rows: int = 0
    target_completeness: float = 0.0
    n_outliers: int = 0
    n_target_hours_imputed: int = 0
    passed: bool = True
    reason: str = ""
    outliers_by_column: Dict[str, int] = field(default_factory=dict)

    def as_row(self) -> Dict[str, object]:
        return {
            "station": self.station,
            "rows": self.n_rows,
            "target_completeness": round(self.target_completeness, 4),
            "outliers_removed": self.n_outliers,
            "target_hours_imputed": self.n_target_hours_imputed,
            "accepted": self.passed,
            "reason": self.reason,
        }


# --------------------------------------------------------------------------- #
def reindex_to_hourly(
    df: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp, *, context: str = ""
) -> pd.DataFrame:
    """Clip to ``[start, end]`` and place the data on a gapless hourly grid.

    Missing hours become all-NaN rows. This must happen before anything counts
    hours: on a sparse index, ``interpolate(limit=3)`` counts three *rows*, which
    can span three days rather than three hours.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise IntegrityError(f"{context}: expected a DatetimeIndex.")
    grid = pd.date_range(start=start, end=end, freq="h")
    clipped = df.loc[(df.index >= grid[0]) & (df.index <= grid[-1])]
    out = clipped.reindex(grid)
    out.index.name = df.index.name or "Date"

    inserted = len(grid) - len(clipped)
    if inserted:
        logger.info(
            "%s: inserted %d missing hourly rows (%.2f%% of the grid).",
            context, inserted, inserted / len(grid) * 100,
        )
    check_monotonic_hourly_index(out, context=f"{context} after reindex")
    return out


# --------------------------------------------------------------------------- #
class IQRScreen:
    """Interquartile-range bounds fitted on training data (paper §2.2).

    A two-phase estimator, deliberately: ``fit`` sees only the training block,
    ``transform`` may be given anything. The asymmetry *is* the leakage control.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
    >>> screen = IQRScreen(k=1.5).fit(train)                    # doctest: +ELLIPSIS
    <...IQRScreen object at ...>
    >>> out, n = screen.transform(pd.DataFrame({"x": [2.0, 500.0]}))
    >>> bool(np.isnan(out["x"].iloc[1])), n
    (True, 1)
    """

    def __init__(self, *, k: float = 1.5, columns: Optional[Sequence[str]] = None) -> None:
        self.k = k
        self.columns = list(columns) if columns else None
        self.bounds_: Dict[str, Tuple[float, float]] = {}

    def fit(self, train_df: pd.DataFrame) -> "IQRScreen":
        candidates = self.columns or [
            c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c])
        ]
        self.bounds_.clear()
        for column in candidates:
            if column not in train_df.columns:
                continue
            series = train_df[column].astype(float)
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr <= 0:
                logger.debug("IQRScreen: %s has a degenerate IQR; not screened.", column)
                continue
            self.bounds_[column] = (float(q1 - self.k * iqr), float(q3 + self.k * iqr))
        logger.info(
            "IQR bounds (k=%.1f) fitted on %d training rows for %d columns.",
            self.k, len(train_df), len(self.bounds_),
        )
        return self

    def transform(
        self, df: pd.DataFrame, *, protect: Sequence[str] = (), context: str = ""
    ) -> Tuple[pd.DataFrame, int]:
        """Void out-of-bound values. ``protect`` columns are never touched.

        Returns ``(screened_frame, n_values_voided)``.
        """
        if not self.bounds_:
            raise RuntimeError("IQRScreen.transform called before fit.")
        out = df.copy()
        protected = set(protect)
        counts: Dict[str, int] = {}

        for column, (low, high) in self.bounds_.items():
            if column not in out.columns or column in protected:
                continue
            series = out[column]
            offending = series.notna() & ((series < low) | (series > high))
            n = int(offending.sum())
            if n:
                counts[column] = n
                out.loc[offending, column] = np.nan

        total = sum(counts.values())
        if total:
            worst = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
            logger.info(
                "%s: IQR screening voided %d values (%.3f%% of cells) across %d columns; "
                "worst: %s",
                context, total, total / max(out.size, 1) * 100, len(counts),
                ", ".join(f"{k}={v}" for k, v in worst),
            )
        return out, total

    def table(self) -> pd.DataFrame:
        """The fitted bounds -- reportable, and worth eyeballing once."""
        return pd.DataFrame(
            [{"column": c, "lower": lo, "upper": hi} for c, (lo, hi) in self.bounds_.items()]
        )


# --------------------------------------------------------------------------- #
def drop_unusable_columns(
    df: pd.DataFrame,
    *,
    drop: Sequence[str] = (),
    protect: Sequence[str] = (),
    max_missing_fraction: float = 0.80,
    context: str = "",
) -> pd.DataFrame:
    """Remove requested, empty and near-empty columns.

    ``protect`` (normally the target) is never dropped, so a station with a
    partly-empty target fails the admission gate with an explicit reason instead
    of silently losing its label column and reporting "target absent".
    """
    out = df.copy()
    protected = set(protect)

    requested = [c for c in drop if c in out.columns and c not in protected]
    if requested:
        out = out.drop(columns=requested)
        logger.debug("%s: dropped configured columns %s", context, requested)

    empty = [c for c in out.columns if c not in protected and out[c].isna().all()]
    if empty:
        out = out.drop(columns=empty)
        logger.info("%s: dropped %d fully-empty columns %s", context, len(empty), empty)

    sparse = [
        c for c in out.columns
        if c not in protected and out[c].isna().mean() > max_missing_fraction
    ]
    if sparse:
        detail = {c: round(float(out[c].isna().mean()), 3) for c in sparse}
        out = out.drop(columns=sparse)
        logger.info(
            "%s: dropped %d columns above the %.0f%% missing threshold: %s",
            context, len(sparse), max_missing_fraction * 100, detail,
        )
    return out


def assess_quality(
    df: pd.DataFrame,
    *,
    station: str,
    target: str,
    min_completeness: float = 0.0,
    min_rows: int = 1,
    report: Optional[QualityReport] = None,
) -> QualityReport:
    """Admission decision, recording the reason either way.

    Completeness is measured on observed values **before** imputation. Measured
    afterwards it would be meaningless: any imputer can make a sparse station
    look complete, and the model would then be trained and scored largely on
    reconstructed labels.
    """
    report = report or QualityReport(station=station)
    report.n_rows = len(df)
    report.target_completeness = (
        float(df[target].notna().mean()) if target in df.columns and len(df) else 0.0
    )

    if target not in df.columns:
        report.passed, report.reason = False, f"target column {target!r} absent"
    elif len(df) < min_rows:
        report.passed = False
        report.reason = f"only {len(df)} rows (need >= {min_rows})"
    elif report.target_completeness < min_completeness:
        report.passed = False
        report.reason = (
            f"target completeness {report.target_completeness:.1%} < required "
            f"{min_completeness:.0%}"
        )
    else:
        report.passed, report.reason = True, ""

    log = logger.info if report.passed else logger.warning
    log(
        "%s: %d rows, target completeness %.1f%% -> %s%s",
        station, report.n_rows, report.target_completeness * 100,
        "ACCEPTED" if report.passed else "REJECTED",
        "" if report.passed else f" ({report.reason})",
    )
    return report
