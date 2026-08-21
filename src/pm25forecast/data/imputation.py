"""k-nearest-neighbours imputation (paper §2.2).

    "Missing values ... were imputed using the k-nearest neighbors (KNN) method.
    To avoid information leakage, the imputation model was trained on the
    training data and then applied to the validation and test sets."

That sentence is the entire contract, and it is enforced structurally: ``fit``
takes the training block, everything else can only be ``transform``ed. The class
cannot be used the wrong way round without writing ``fit`` twice, which is
visible in review.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from ..logging_utils import get_logger
from ..validation import IntegrityError

logger = get_logger(__name__)

__all__ = ["TrainFittedKNNImputer", "interpolate_short_gaps"]


def interpolate_short_gaps(
    df: pd.DataFrame, *, limit: int = 3, method: str = "pchip", context: str = ""
) -> pd.DataFrame:
    """Fill gaps of at most ``limit`` hours by interpolation (**beyond the paper**).

    Disabled by default. When enabled it runs *before* KNN, on the reasoning that
    a one-hour dropout is better served by its immediate neighbours than by a
    multivariate donor from another day. PCHIP is shape-preserving and will not
    overshoot into negative concentrations the way a cubic spline can.

    Note that this uses the value at the far edge of the gap, a bounded
    look-ahead of at most ``limit`` hours -- standard for offline gap filling,
    but not available to a real-time system.
    """
    if limit <= 0:
        return df
    out = df.copy()
    filled = 0
    for column in out.columns:
        if not pd.api.types.is_numeric_dtype(out[column]) or not out[column].isna().any():
            continue
        before = int(out[column].isna().sum())
        try:
            out[column] = out[column].interpolate(method=method, limit=limit, limit_direction="both")
        except (ValueError, ImportError):
            out[column] = out[column].interpolate(method="linear", limit=limit, limit_direction="both")
        filled += before - int(out[column].isna().sum())
    if filled:
        logger.info("%s: %s interpolation (<=%dh) filled %d values.", context, method, limit, filled)
    return out


class TrainFittedKNNImputer:
    """KNN imputation fitted on the training block only (paper §2.2).

    Examples
    --------
    >>> imputer = TrainFittedKNNImputer(n_neighbors=5)      # doctest: +SKIP
    >>> imputer.fit(train_df)                                # doctest: +SKIP
    >>> filled_test = imputer.transform(test_df)             # doctest: +SKIP
    """

    def __init__(self, *, n_neighbors: int = 5, weights: str = "uniform") -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._imputer: Optional[KNNImputer] = None
        self._columns: List[str] = []
        self._fit_span: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None

    def fit(self, train_df: pd.DataFrame) -> "TrainFittedKNNImputer":
        numeric = train_df.select_dtypes(include=[np.number])
        if numeric.empty:
            raise IntegrityError("TrainFittedKNNImputer.fit: no numeric columns.")
        all_missing = [c for c in numeric.columns if numeric[c].isna().all()]
        if all_missing:
            raise IntegrityError(
                f"TrainFittedKNNImputer.fit: columns {all_missing} are entirely missing in the "
                f"training block, so KNN has nothing to learn from. Drop them first."
            )
        self._columns = list(numeric.columns)
        self._imputer = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)
        # Fitted on a bare ndarray: sklearn binds feature names to the first array
        # it sees, and mixing named and unnamed inputs later emits warnings that
        # drown the pipeline's own diagnostics.
        self._imputer.fit(numeric.to_numpy(dtype=float))
        if isinstance(train_df.index, pd.DatetimeIndex) and len(train_df):
            self._fit_span = (train_df.index.min(), train_df.index.max())
        logger.info(
            "KNN imputer (k=%d) fitted on %d training rows x %d columns%s.",
            self.n_neighbors, len(numeric), len(self._columns),
            f", {self._fit_span[0]} -> {self._fit_span[1]}" if self._fit_span else "",
        )
        return self

    def transform(self, df: pd.DataFrame, *, context: str = "") -> pd.DataFrame:
        if self._imputer is None:
            raise RuntimeError("TrainFittedKNNImputer.transform called before fit.")
        if df.empty:
            return df.copy()

        missing = [c for c in self._columns if c not in df.columns]
        if missing:
            raise IntegrityError(
                f"{context}: columns absent from the frame: {missing}. Fit and transform must "
                f"see the same schema, or the imputer silently matches the wrong variables."
            )
        block = df[self._columns]
        before = int(block.isna().sum().sum())
        if before == 0:
            return df.copy()

        filled = pd.DataFrame(
            self._imputer.transform(block.to_numpy(dtype=float)),
            columns=self._columns, index=block.index,
        )
        out = df.copy()
        out[self._columns] = filled
        after = int(out[self._columns].isna().sum().sum())
        logger.info(
            "%s: KNN filled %d of %d missing values; %d cells remain missing.",
            context, before - after, before, after,
        )
        return out

    def assert_fitted_on(self, span: Tuple[pd.Timestamp, pd.Timestamp]) -> None:
        """Confirm from recorded state that only training rows were used."""
        if self._fit_span is None:
            raise IntegrityError("Imputer has no recorded fitting span.")
        start, end = span
        if self._fit_span[0] < start or self._fit_span[1] > end:
            raise IntegrityError(
                f"Imputer was fitted on {self._fit_span[0]} -> {self._fit_span[1]}, which "
                f"extends outside the training block [{start}, {end}]. This is leakage."
            )
