"""Stage 4 -- score everything on the held-out 2024 test year (paper §4.4, §5).

Scored per station: the four base learners, all six ensembles, and any optional
baselines. Reported at three levels:

* **hourly** -- did the model track the diurnal cycle, or reach a good daily mean
  by averaging two opposite errors;
* **Day 1** (lead 1-24 h) and **Day 2** (lead 25-48 h) daily means -- the paper's
  headline, and the level at which predictability decay becomes visible;
* **strategy** -- the three families of §4.2 compared directly, which is the
  comparison the paper is actually making.

Nothing here fits anything. If a line in this module calls ``.fit``, it is a bug.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from ..baselines import build_baselines
from ..config import Config
from ..logging_utils import get_logger, section
from ..metrics import metrics_table, summarise_forecast
from ..models.ensembles import ENSEMBLE_STRATEGY
from ..validation import IntegrityError
from .preprocess import load_station_table
from .train_ensembles import (
    ENSEMBLE_META,
    _hours_of,
    load_member_predictions,
    predict_with_ensemble,
)

logger = get_logger(__name__)

__all__ = ["evaluate_station", "run_evaluation"]


def evaluate_station(slug: str, cfg: Config) -> Optional[pd.DataFrame]:
    """Score every forecaster for one station."""
    payload, base_meta = load_member_predictions(cfg, slug)
    station = str(base_meta["station_name"])
    member_names: List[str] = list(base_meta["model_names"])

    members = payload["test_members"].astype(np.float64)
    y = payload["test_y"].astype(np.float64)
    context = payload["test_context"].astype(np.float64)
    hours = _hours_of(payload["test_target_times"])
    if len(y) == 0:
        logger.warning("%s: empty test block.", station)
        return None

    n, horizon, _ = members.shape
    context_flat = context.reshape(n * horizon, -1)
    clip = cfg.evaluation.prediction_clip

    forecasts: Dict[str, np.ndarray] = {
        name: members[:, :, index] for index, name in enumerate(member_names)
    }
    strategy: Dict[str, str] = {name: "Base Learner" for name in member_names}

    # -- ensembles ----------------------------------------------------------- #
    ensemble_dir = cfg.ensemble_dir / slug
    if (ensemble_dir / ENSEMBLE_META).exists():
        for name in cfg.ensembles.names:
            path = ensemble_dir / f"{name}.joblib"
            if not path.exists():
                logger.warning("%s: ensemble %s was not fitted; skipped.", station, name)
                continue
            ensemble = joblib.load(path)
            forecasts[name] = predict_with_ensemble(ensemble, members, context_flat, hours)
            strategy[name] = ENSEMBLE_STRATEGY.get(name, "unknown")
    else:
        logger.warning("%s: no ensembles found; scoring base learners only.", station)

    # -- optional baselines (beyond the paper) -------------------------------- #
    if cfg.evaluation.baselines:
        table, _, _ = load_station_table(cfg, slug)
        train_start, train_end = base_meta.get("train_span", [None, None]) if isinstance(
            base_meta.get("train_span"), list
        ) else (None, None)
        train_target = (
            table.loc[train_start:train_end, cfg.data.target]
            if train_start
            else table.loc[cfg.split.development[0] : cfg.split.development[1], cfg.data.target]
        )
        anchors = pd.to_datetime(payload["test_anchors"].tolist())
        input_times = np.stack([
            pd.date_range(a, periods=cfg.windows.input_hours, freq="h").to_numpy()
            for a in anchors
        ])
        target_times = np.array(
            [[np.datetime64(t) for t in row] for row in payload["test_target_times"]]
        )
        extra = build_baselines(
            cfg.evaluation.baselines,
            train_target=train_target,
            history=payload["test_history"],
            input_times=input_times,
            target_times=target_times,
            horizon=horizon,
            clip=clip,
        )
        forecasts.update(extra)
        strategy.update({k: "Baseline" for k in extra})

    # -- physical bounds, applied to EVERY forecaster ------------------------- #
    # A negative PM2.5 concentration is impossible for any model, so the bound is
    # a property of the quantity and must be applied uniformly. Clipping only the
    # ensembles would hand them an advantage over the base learners they are
    # compared against -- small, but the same order as the margin being reported.
    if clip:
        forecasts = {
            name: np.clip(prediction, clip[0], clip[1])
            for name, prediction in forecasts.items()
        }

    # -- score ---------------------------------------------------------------- #
    rows: List[Dict[str, object]] = []
    for name, prediction in forecasts.items():
        row: Dict[str, object] = {
            "station": station, "slug": slug,
            "strategy": strategy.get(name, "unknown"), "model": name,
        }
        row.update(
            summarise_forecast(
                y, prediction,
                min_hours_per_day=cfg.evaluation.min_hours_per_day,
                threshold=cfg.evaluation.peak_threshold_ugm3,
            )
        )
        rows.append(row)
    frame = pd.DataFrame(rows)

    # -- the comparison the paper's claim rests on ---------------------------- #
    best_member = frame.loc[frame["model"].isin(member_names), "day1_RMSE"].min()
    ensemble_rows = frame[~frame["model"].isin(member_names) & (frame["strategy"] != "Baseline")]
    if not ensemble_rows.empty:
        best_ensemble_row = ensemble_rows.loc[ensemble_rows["day1_RMSE"].idxmin()]
        logger.info(
            "%s: best base learner Day-1 RMSE %.3f | best ensemble %s (%s) %.3f (%+.1f%%).",
            station, best_member, best_ensemble_row["model"], best_ensemble_row["strategy"],
            best_ensemble_row["day1_RMSE"],
            (best_ensemble_row["day1_RMSE"] / best_member - 1) * 100,
        )
        if best_ensemble_row["day1_RMSE"] >= best_member:
            logger.warning(
                "%s: no ensemble beats the best single base learner on Day 1. Report it as "
                "it is -- that is a result, not a bug.", station,
            )
    frame["skill_vs_best_member_day1"] = 1.0 - frame["day1_RMSE"] / best_member

    frame.to_csv(cfg.results_dir / f"metrics_{slug}.csv", index=False, encoding="utf-8-sig")
    _write_predictions(cfg, slug, payload, y, forecasts)
    return frame


def _write_predictions(cfg: Config, slug: str, payload, y: np.ndarray, forecasts) -> None:
    """Long-format forecasts so any figure can be redrawn without retraining."""
    times = np.array([[np.datetime64(t) for t in row] for row in payload["test_target_times"]])
    horizon = y.shape[1]
    records = {
        "timestamp": pd.to_datetime(times.ravel()),
        "lead_hour": np.tile(np.arange(1, horizon + 1), y.shape[0]),
        "forecast_day": np.tile(np.arange(horizon) // 24 + 1, y.shape[0]),
        "observed": y.ravel(),
    }
    for name, values in forecasts.items():
        records[name] = np.asarray(values, dtype=float).ravel()
    pd.DataFrame(records).to_parquet(
        cfg.results_dir / f"predictions_{slug}.parquet", index=False
    )


# --------------------------------------------------------------------------- #
def run_evaluation(cfg: Config) -> pd.DataFrame:
    """Evaluate every station and write the combined, city-level and strategy tables."""
    section(logger, "STAGE 4 / EVALUATION ON THE HELD-OUT TEST YEAR  (paper 4.4)")

    slugs = sorted(
        p.name for p in cfg.artifacts_dir.iterdir()
        if p.is_dir() and (p / "member_predictions.npz").exists()
    )
    if not slugs:
        raise FileNotFoundError(f"No trained stations in {cfg.artifacts_dir}.")

    frames: List[pd.DataFrame] = []
    for slug in slugs:
        try:
            frame = evaluate_station(slug, cfg)
        except (IntegrityError, ValueError, KeyError, FileNotFoundError) as exc:
            logger.error("%s: evaluation failed -- %s", slug, exc)
            continue
        if frame is not None:
            frames.append(frame)

    if not frames:
        raise IntegrityError("No station could be evaluated.")

    combined = metrics_table(pd.concat(frames, ignore_index=True).to_dict("records"))
    combined.to_csv(
        cfg.results_dir / "metrics_all_stations.csv", index=False, encoding="utf-8-sig"
    )

    numeric = combined.select_dtypes(include=[np.number]).columns
    city = combined.groupby(["strategy", "model"])[list(numeric)].mean(numeric_only=True).reset_index()
    city["n_stations"] = combined.groupby(["strategy", "model"]).size().to_numpy()
    city = city.sort_values("day1_RMSE").reset_index(drop=True)
    city.to_csv(cfg.results_dir / "metrics_city_mean.csv", index=False, encoding="utf-8-sig")

    # Strategy-level comparison -- the actual question of paper 4.2.
    by_strategy = (
        city.groupby("strategy")[["day1_RMSE", "day2_RMSE", "day1_R", "day1_IOA"]]
        .mean()
        .sort_values("day1_RMSE")
        .reset_index()
    )
    by_strategy.to_csv(
        cfg.results_dir / "metrics_by_strategy.csv", index=False, encoding="utf-8-sig"
    )

    headline = [
        c for c in ("strategy", "model", "n_stations", "day1_RMSE", "day1_MAE", "day1_R",
                    "day1_IOA", "day1_Unc_percent", "day2_RMSE", "day2_R")
        if c in city.columns
    ]
    logger.info(
        "City-mean summary (lower Day-1 RMSE is better):\n%s",
        city[headline].to_string(index=False, float_format=lambda v: f"{v:.3f}"),
    )
    logger.info(
        "By strategy:\n%s",
        by_strategy.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
    )
    logger.info("Results written to %s", cfg.results_dir)
    return combined
