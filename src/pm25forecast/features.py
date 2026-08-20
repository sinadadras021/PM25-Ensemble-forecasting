r"""Feature construction (paper §2.2).

Everything here is a deterministic function of the input frame -- nothing is
fitted, so this stage cannot leak. Anything that *is* fitted (screens, imputers,
scalers) lives elsewhere.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "TARGET_INPUT_SUFFIX",
    "target_input_name",
    "wind_to_components",
    "add_cyclical_time_features",
    "hour_offset_grid",
    "build_feature_frame",
    "select_feature_columns",
]

_CYCLES: Dict[str, float] = {"hour": 24.0, "dayofweek": 7.0, "doy": 365.25}

#: Suffix marking the gap-filled copy of the target used as an encoder input.
TARGET_INPUT_SUFFIX = "_input"


def target_input_name(target: str) -> str:
    """Column name of the gap-filled input copy of ``target``.

    The label and the input history are deliberately different objects:

    * the **label** may be observations only -- scoring against reconstructed
      values would measure the imputer rather than the forecaster;
    * the **input history** is always gap-filled, because a window must be
      complete to exist at all, and one missing input hour would otherwise cost
      the 96 hours of window around it.

    Conflating the two is the reason this distinction lives in code rather than
    in a comment.

    >>> target_input_name("PM 2.5 ug/m3")
    'PM 2.5 ug/m3_input'
    """
    return f"{target}{TARGET_INPUT_SUFFIX}"


def wind_to_components(
    df: pd.DataFrame,
    *,
    speed_column: str = "wind_speed",
    direction_column: str = "wind_direction",
    context: str = "",
) -> pd.DataFrame:
    r"""Replace speed/direction by U and V components (paper Eqs. 1-2).

    .. math::

        U = -\,\mathrm{WS}\sin\!\left(\tfrac{\pi}{180}\,\mathrm{WD}\right),
        \qquad
        V = -\,\mathrm{WS}\cos\!\left(\tfrac{\pi}{180}\,\mathrm{WD}\right)

    Wind direction is *periodic*: 359 deg and 1 deg are adjacent in the world and
    358 units apart on a number line. A network fed the raw angle must spend
    capacity learning that discontinuity, and optimisation through it is badly
    conditioned. The (U, V) pair is continuous everywhere and linear in the
    physically meaningful quantity -- the transport vector.

    A no-op when the archive already stores U and V, as the reference dataset
    does.
    """
    if speed_column not in df.columns or direction_column not in df.columns:
        logger.debug(
            "%s: %r/%r not found; assuming U and V are already present.",
            context, speed_column, direction_column,
        )
        return df

    out = df.copy()
    theta = np.radians(out[direction_column].to_numpy(dtype=float))
    speed = out[speed_column].to_numpy(dtype=float)
    out["U"] = -speed * np.sin(theta)
    out["V"] = -speed * np.cos(theta)
    out = out.drop(columns=[speed_column, direction_column])
    logger.info("%s: converted wind speed/direction to U and V components.", context)
    return out


def add_cyclical_time_features(
    df: pd.DataFrame, which: Sequence[str] = ("hour",)
) -> pd.DataFrame:
    r"""Sin/cos encodings of clock position (paper Eqs. 3-4).

    .. math::

        \mathrm{Hour\_sin} = \sin\frac{2\pi\,\mathrm{Hour}}{24},
        \qquad
        \mathrm{Hour\_cos} = \cos\frac{2\pi\,\mathrm{Hour}}{24}

    Hour 23 and hour 00 are one hour apart in the world and 23 units apart on a
    number line; the sin/cos pair restores the circular geometry the atmosphere
    actually has. For PM2.5 this matters because the diurnal cycle is driven by
    the boundary layer and by traffic, both of which wrap at midnight.

    >>> idx = pd.date_range("2024-06-01", periods=3, freq="h")
    >>> sorted(add_cyclical_time_features(pd.DataFrame(index=idx)).columns)
    ['Hour_cos', 'Hour_sin']
    """
    out = df.copy()
    index = out.index
    labels = {"hour": "Hour", "dayofweek": "Weekday", "doy": "DOY"}
    for name in which:
        if name not in _CYCLES:
            raise ValueError(f"Unsupported cyclical feature {name!r}; known: {sorted(_CYCLES)}")
        value = {
            "hour": index.hour, "dayofweek": index.dayofweek, "doy": index.dayofyear,
        }[name].to_numpy(dtype=float)
        angle = 2.0 * np.pi * value / _CYCLES[name]
        out[f"{labels[name]}_sin"] = np.sin(angle)
        out[f"{labels[name]}_cos"] = np.cos(angle)
    return out


def hour_offset_grid(n_windows: int, horizon: int) -> np.ndarray:
    """``Hour_offset`` for every forecast step (paper §2.2, §4.2).

    Integer position within the forecast window, 0 .. horizon-1. This is what
    lets a meta-learner model **lead-time decay** -- to lean on a different member
    at hour 40 than at hour 2 -- which no fixed linear blend can express.

    >>> hour_offset_grid(2, 4).tolist()
    [[0, 1, 2, 3], [0, 1, 2, 3]]
    """
    return np.tile(np.arange(horizon, dtype=int), (n_windows, 1))


def build_feature_frame(df: pd.DataFrame, cfg, *, context: str = "") -> pd.DataFrame:
    """Apply every unfitted transform in the canonical order."""
    out = df
    if cfg.features.wind_components:
        out = wind_to_components(
            out,
            speed_column=cfg.features.wind_speed_column,
            direction_column=cfg.features.wind_direction_column,
            context=context,
        )
    cyclical = list(cfg.features.cyclical)
    if cfg.features.add_day_of_year and "doy" not in cyclical:
        cyclical.append("doy")
    out = add_cyclical_time_features(out, cyclical)
    return out


def select_feature_columns(
    df: pd.DataFrame, *, target: str, exclude: Sequence[str] = ()
) -> List[str]:
    """Ordered encoder inputs.

    The target's gap-filled **input copy** is included: the previous 48 hours of
    PM2.5 are available at issue time and are the single most informative
    predictor of the next 48. The raw label column is excluded, so the encoder
    can never read an unobserved value that the metrics would then score
    against.

    Sorted, so feature order is stable across runs and machines. A saved scaler
    and a saved model must agree on column order forever; a silent reordering
    produces plausible-looking nonsense rather than an error.
    """
    excluded = set(exclude) | {target}
    columns = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    history = target_input_name(target)
    if history not in columns:
        raise ValueError(
            f"select_feature_columns: {history!r} is missing. The gap-filled target history "
            f"must be built before feature selection."
        )
    if len(columns) < 2:
        raise ValueError(
            f"select_feature_columns produced only {columns}; a univariate input defeats the "
            f"multivariate premise of the method."
        )
    return sorted(columns)
