"""Leakage-free scaling.

One rule, enforced by construction rather than by convention: **a scaler is
fitted on the training block and on nothing else.** :class:`SplitScaler` takes
the training slice in its constructor, records the timestamps it saw, and exposes
that record so :func:`~pm25forecast.validation.check_scaler_fitted_on_train_only`
can verify the claim at runtime.

A note on MinMax versus Robust for a forecasting problem. MinMax maps the
training range onto ``[0, 1]``; an unprecedented test-set extreme therefore maps
above 1 and the network must extrapolate. That is *honest* -- it is exactly the
situation an operational model faces -- but it is also why the peak-heavy loss
and the optional ``clip_scaled`` guard exist. RobustScaler (median/IQR) degrades
more gracefully in the tails and is available via ``method="robust"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .logging_utils import get_logger
from .validation import IntegrityError, check_scaler_fitted_on_train_only

logger = get_logger(__name__)

__all__ = ["make_scaler", "SplitScaler"]

_SCALERS = {"minmax": MinMaxScaler, "robust": RobustScaler, "standard": StandardScaler}


def make_scaler(method: str):
    """Instantiate a scikit-learn scaler by short name."""
    try:
        return _SCALERS[method]()
    except KeyError as exc:  # pragma: no cover - guarded by config validation
        raise ValueError(f"Unknown scaling method {method!r}; known: {sorted(_SCALERS)}") from exc


class SplitScaler:
    """Paired feature/target scalers with a provable training-only fit.

    Parameters
    ----------
    feature_columns:
        Fixed, ordered list. Persisted with the scaler so that a model reloaded
        months later cannot be fed columns in a different order -- a failure mode
        that produces plausible-looking nonsense rather than an exception.
    clip:
        Optional ``(low, high)`` applied after transform. Guards against a single
        pathological test value driving a recurrent layer into saturation.

    Examples
    --------
    >>> scaler = SplitScaler(["a", "b"], "y", method="minmax")     # doctest: +SKIP
    >>> scaler.fit(train_df)                                        # doctest: +SKIP
    >>> scaled = scaler.transform(any_df)                           # doctest: +SKIP
    """

    def __init__(
        self,
        feature_columns: Sequence[str],
        target_column: str,
        *,
        method: str = "minmax",
        clip: Optional[Tuple[float, float]] = None,
    ) -> None:
        if target_column in feature_columns:
            logger.debug(
                "Target %r is also an input feature (autoregressive setup); it will be scaled "
                "twice, once by scaler_X for the encoder and once by scaler_y for the labels.",
                target_column,
            )
        self.feature_columns: List[str] = list(feature_columns)
        self.target_column = target_column
        self.method = method
        self.clip = tuple(clip) if clip is not None else None
        self.scaler_X = make_scaler(method)
        self.scaler_y = make_scaler(method)
        self._fit_index: Optional[pd.DatetimeIndex] = None

    # -- fit ---------------------------------------------------------------- #
    def fit(self, train_df: pd.DataFrame, *, min_rows: int = 1) -> "SplitScaler":
        """Fit on rows of ``train_df`` that have both complete features and a target.

        Rows with a missing target are excluded from the *target* scaler's fit so
        that the ``y`` range reflects observations only.
        """
        missing = [c for c in self.feature_columns if c not in train_df.columns]
        if missing:
            raise IntegrityError(f"SplitScaler.fit: columns absent from training frame: {missing}")
        if self.target_column not in train_df.columns:
            raise IntegrityError(
                f"SplitScaler.fit: target {self.target_column!r} absent from training frame."
            )

        usable = train_df.dropna(subset=[self.target_column] + self.feature_columns)
        if len(usable) < min_rows:
            raise IntegrityError(
                f"SplitScaler.fit: only {len(usable)} complete training rows "
                f"(need >= {min_rows}). The station cannot support a trustworthy scaler."
            )

        self.scaler_X.fit(usable[self.feature_columns].to_numpy(dtype=float))
        self.scaler_y.fit(usable[[self.target_column]].to_numpy(dtype=float))
        self._fit_index = pd.DatetimeIndex(usable.index)
        logger.info(
            "SplitScaler(%s) fitted on %d rows, %d features, %s -> %s",
            self.method, len(usable), len(self.feature_columns),
            self._fit_index.min(), self._fit_index.max(),
        )
        return self

    def assert_train_only(self, train_span: Tuple[pd.Timestamp, pd.Timestamp]) -> None:
        """Re-derive, from recorded state, that no non-training row was used."""
        if self._fit_index is None:
            raise IntegrityError("SplitScaler.assert_train_only called before fit.")
        check_scaler_fitted_on_train_only(
            self.scaler_X, self._fit_index, train_span, context="SplitScaler"
        )

    # -- transform ---------------------------------------------------------- #
    def transform(self, df: pd.DataFrame, *, with_target: bool = True) -> pd.DataFrame:
        """Scale features (and optionally the target) of an arbitrary slice.

        NaNs pass through untouched -- they mark gaps that the window builder must
        still be able to see and reject.
        """
        if df.empty:
            return df.copy()
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise IntegrityError(f"SplitScaler.transform: columns absent: {missing}")

        values = self.scaler_X.transform(df[self.feature_columns].to_numpy(dtype=float))
        if self.clip is not None:
            values = np.clip(values, self.clip[0], self.clip[1])
        out = pd.DataFrame(values, columns=self.feature_columns, index=df.index)

        if with_target:
            if self.target_column not in df.columns:
                raise IntegrityError(
                    f"SplitScaler.transform: target {self.target_column!r} absent."
                )
            out["Target"] = self.scaler_y.transform(
                df[[self.target_column]].to_numpy(dtype=float)
            )
        return out

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        """Map scaled predictions back to ppb, preserving shape and NaNs."""
        arr = np.asarray(values, dtype=float)
        shape = arr.shape
        flat = arr.reshape(-1, 1)
        nan_mask = np.isnan(flat)
        restored = self.scaler_y.inverse_transform(np.where(nan_mask, 0.0, flat))
        restored[nan_mask] = np.nan
        return restored.reshape(shape)

    # -- persistence -------------------------------------------------------- #
    def save(self, directory: str | Path, *, prefix: str = "scaler") -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "method": self.method,
            "clip": self.clip,
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "fit_start": None if self._fit_index is None else str(self._fit_index.min()),
            "fit_end": None if self._fit_index is None else str(self._fit_index.max()),
            "fit_rows": 0 if self._fit_index is None else len(self._fit_index),
        }
        destination = path / f"{prefix}.joblib"
        joblib.dump(payload, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "SplitScaler":
        payload = joblib.load(Path(path))
        obj = cls(
            payload["feature_columns"],
            payload["target_column"],
            method=payload["method"],
            clip=payload["clip"],
        )
        obj.scaler_X = payload["scaler_X"]
        obj.scaler_y = payload["scaler_y"]
        return obj
