"""Stage 5 -- leave-one-member-out ablation (paper §5.5).

The question: does each base learner earn its place? An ensemble of four models
is only justified if removing any one measurably hurts. If dropping a member
leaves the meta-learner unchanged, that member costs training time and inference
latency for nothing, and a reviewer is entitled to ask why it is there.

Procedure, for each member *m* and each meta-learner:

1. remove *m* from the member block;
2. **refit** the meta-learner on the reduced input -- not merely zero the column,
   which would leave a model expecting a feature that is now constant;
3. score on the test block and report the change in Day-1 RMSE.

Refitting is what makes the number interpretable: the reduced ensemble is the
*best available* ensemble without that member, so the degradation measures
irreplaceable information rather than a broken input shape.

**Why the split stays chronological.** The ablation reuses the same
chronological validation/test split as the main run rather than a shuffled
``KFold``. On an hourly series consecutive rows are near-duplicates, so random
folds place each row's neighbours on the other side of the boundary and the
resulting error measures interpolation rather than forecasting.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger, section
from ..metrics import continuous_metrics, split_days
from ..models.ensembles import ENSEMBLE_STRATEGY, build_ensemble
from ..seeding import set_global_seed
from ..validation import IntegrityError
from .train_ensembles import (
    _fit_one,
    _hours_of,
    load_member_predictions,
    predict_with_ensemble,
)

logger = get_logger(__name__)

__all__ = ["ablation_for_station", "run_ablation"]


def _score(y: np.ndarray, prediction: np.ndarray, cfg: Config) -> Dict[str, float]:
    truth = split_days(y, min_hours=cfg.evaluation.min_hours_per_day)
    pred = split_days(prediction, min_hours=cfg.evaluation.min_hours_per_day)
    out: Dict[str, float] = {}
    for label in truth:
        out.update(continuous_metrics(truth[label], pred[label], prefix=f"{label}_"))
    out.update(continuous_metrics(y, prediction, prefix="hourly_"))
    return out


def ablation_for_station(
    slug: str, cfg: Config, ensembles: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """Leave-one-member-out table for one station."""
    payload, base_meta = load_member_predictions(cfg, slug)
    station = str(base_meta["station_name"])
    member_names: List[str] = list(base_meta["model_names"])
    if len(member_names) < 3:
        logger.warning(
            "%s: only %d members; a leave-one-out study needs at least 3 to be meaningful.",
            station, len(member_names),
        )
        return None

    names = ensembles or [n for n in cfg.ensembles.names if n not in {"sa"}]
    if not names:
        return None

    val_members = payload["val_members"].astype(np.float64)
    val_y = payload["val_y"].astype(np.float64)
    val_context = payload["val_context"].astype(np.float64)
    val_hours = _hours_of(payload["val_target_times"])

    test_members = payload["test_members"].astype(np.float64)
    test_y = payload["test_y"].astype(np.float64)
    test_context = payload["test_context"].astype(np.float64)
    test_hours = _hours_of(payload["test_target_times"])

    val_flat = val_context.reshape(val_members.shape[0] * val_members.shape[1], -1)
    test_flat = test_context.reshape(test_members.shape[0] * test_members.shape[1], -1)

    logger.info("=" * 78)
    logger.info("Ablation | %s | %d members x %d ensembles", station, len(member_names), len(names))

    rows: List[Dict[str, object]] = []
    for ensemble_name in names:
        baseline_score: Optional[float] = None

        for excluded in ["(none)"] + member_names:
            keep = [m for m in member_names if m != excluded]
            index = [member_names.index(m) for m in keep]
            v_members, t_members = val_members[:, :, index], test_members[:, :, index]

            set_global_seed(cfg.seed, deterministic=cfg.deterministic)
            ensemble = build_ensemble(ensemble_name, cfg, seed=cfg.seed)
            try:
                _fit_one(
                    ensemble_name, ensemble, v_members, val_y, val_flat, val_hours,
                    keep, keep + list(base_meta.get("context_columns", [])),
                )
                prediction = predict_with_ensemble(ensemble, t_members, test_flat, test_hours)
            except (ValueError, RuntimeError, ImportError) as exc:
                logger.error(
                    "%s: %s without %s could not be fitted -- %s",
                    station, ensemble_name, excluded, exc,
                )
                continue
            finally:
                del ensemble
                gc.collect()

            stats = _score(test_y, prediction, cfg)
            if excluded == "(none)":
                baseline_score = stats["day1_RMSE"]
            delta = (
                stats["day1_RMSE"] - baseline_score
                if baseline_score is not None and np.isfinite(baseline_score)
                else float("nan")
            )
            rows.append({
                "station": station,
                "ensemble": ensemble_name,
                "strategy": ENSEMBLE_STRATEGY.get(ensemble_name, "unknown"),
                "excluded": excluded,
                "n_members": len(keep),
                **stats,
                "delta_day1_RMSE": delta,
                "delta_day1_RMSE_percent": (
                    delta / baseline_score * 100 if baseline_score else float("nan")
                ),
            })
            if excluded != "(none)":
                logger.info(
                    "%s: %-9s without %-8s Day-1 RMSE %.3f (%+.2f%%)",
                    station, ensemble_name, excluded, stats["day1_RMSE"],
                    rows[-1]["delta_day1_RMSE_percent"],
                )

    if not rows:
        return None
    frame = pd.DataFrame(rows)
    frame.to_csv(cfg.results_dir / f"ablation_{slug}.csv", index=False, encoding="utf-8-sig")
    return frame


def run_ablation(cfg: Config) -> pd.DataFrame:
    """Leave-one-member-out study across every trained station (paper §5.5)."""
    section(logger, "STAGE 5 / ABLATION STUDY  (paper 5.5)")

    slugs = sorted(
        p.name for p in cfg.artifacts_dir.iterdir()
        if p.is_dir() and (p / "member_predictions.npz").exists()
    )
    frames: List[pd.DataFrame] = []
    for slug in slugs:
        try:
            frame = ablation_for_station(slug, cfg)
        except (IntegrityError, ValueError, KeyError, FileNotFoundError) as exc:
            logger.error("%s: ablation failed -- %s", slug, exc)
            continue
        if frame is not None:
            frames.append(frame)

    if not frames:
        logger.warning("No station produced an ablation table.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(
        cfg.results_dir / "ablation_all_stations.csv", index=False, encoding="utf-8-sig"
    )

    average = (
        combined[combined["excluded"] != "(none)"]
        .groupby("excluded")[["delta_day1_RMSE", "delta_day1_RMSE_percent"]]
        .mean()
        .sort_values("delta_day1_RMSE", ascending=False)
        .reset_index()
    )
    average.to_csv(cfg.results_dir / "ablation_city_mean.csv", index=False, encoding="utf-8-sig")
    logger.info(
        "Mean Day-1 RMSE penalty for removing each base learner "
        "(higher = more indispensable):\n%s",
        average.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
    )
    if not average.empty and average.iloc[-1]["delta_day1_RMSE_percent"] < 0:
        logger.warning(
            "Removing %s *improved* the ensembles on average. That member is contributing "
            "noise, and its inclusion needs a justification beyond 'more models is better'.",
            average.iloc[-1]["excluded"],
        )
    return combined
