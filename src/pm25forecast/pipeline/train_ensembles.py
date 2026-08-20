"""Stage 3 -- fit the six ensembles (paper §4.2).

Every ensemble is fitted on the **validation** block, where the base members'
predictions are genuinely out-of-sample, and applied once to the test year.

Fitting a stack on in-sample base predictions is the classic failure of stacked
generalisation: the members look near-perfect on data they memorised, the
meta-learner learns to trust whichever one overfit hardest, and the structure
collapses on anything new. Because that mistake is invisible in the training
metrics, :func:`_assert_fitted_on_validation` re-derives the provenance of the
arrays being used and refuses to continue if they came from the test block.

Artefacts per station::

    03_ensembles/<slug>/
        sa.joblib, pwe.joblib, mlp_meta.joblib, ...
        ensemble_meta.json          <- weights, selector accuracy, H-MOS table
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger, section
from ..metrics import continuous_metrics
from ..models.ensembles import (
    AVAILABLE_ENSEMBLES,
    ENSEMBLE_STRATEGY,
    HourlyMOS,
    MLPMetaLearner,
    SmartModelSelector,
    XGBoostMetaLearner,
    build_ensemble,
)
from ..seeding import set_global_seed
from ..validation import IntegrityError, check_finite

logger = get_logger(__name__)

__all__ = ["load_member_predictions", "fit_ensembles_for_station", "run_ensemble_training",
           "predict_with_ensemble", "MEMBER_FILE"]

MEMBER_FILE = "member_predictions.npz"
ENSEMBLE_META = "ensemble_meta.json"


def load_member_predictions(cfg: Config, slug: str) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Load stage-2 outputs for one station."""
    model_dir = cfg.artifacts_dir / slug
    array_path = model_dir / MEMBER_FILE
    meta_path = model_dir / "base_model_meta.json"
    if not (array_path.exists() and meta_path.exists()):
        raise FileNotFoundError(
            f"{slug}: stage-2 artefacts missing at {model_dir}. Run train-base first."
        )
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    with np.load(array_path, allow_pickle=False) as arrays:
        payload = {key: arrays[key] for key in arrays.files}
    return payload, meta


def _assert_fitted_on_validation(payload: Dict[str, np.ndarray], meta: Dict[str, object]) -> None:
    """Confirm the arrays about to be fitted on really are the validation block.

    Cheap, and it closes the one edit that would quietly invalidate every number
    in stage 4: swapping ``val_*`` for ``test_*`` in the calls below.
    """
    n_expected = int(meta["n_val_windows"])
    n_actual = int(payload["val_members"].shape[0])
    if n_actual != n_expected:
        raise IntegrityError(
            f"Validation arrays hold {n_actual} windows but stage 2 recorded {n_expected}. "
            f"The ensembles must be fitted on the validation block, never on test."
        )
    val_end = pd.to_datetime(payload["val_anchors"].tolist()).max()
    test_start = pd.to_datetime(payload["test_anchors"].tolist()).min()
    if val_end >= test_start:
        raise IntegrityError(
            f"The validation block ends at {val_end}, at or after the test block start "
            f"{test_start}. The blocks are not chronologically disjoint."
        )


def _hours_of(target_times: np.ndarray) -> np.ndarray:
    """Clock hour of every forecast step -- H-MOS needs it (paper Eq. 11)."""
    times = pd.DatetimeIndex(pd.to_datetime(np.asarray(target_times).ravel()))
    return times.hour.to_numpy().reshape(np.asarray(target_times).shape)


def _fit_one(
    name: str,
    ensemble,
    members: np.ndarray,
    y: np.ndarray,
    context: np.ndarray,
    hours: np.ndarray,
    member_names: List[str],
    feature_names: List[str],
):
    """Dispatch to each ensemble's own fitting signature."""
    if isinstance(ensemble, HourlyMOS):
        return ensemble.fit(members, y, hours=hours)
    if isinstance(ensemble, SmartModelSelector):
        return ensemble.fit(members, y, context=context, member_names=member_names)
    if isinstance(ensemble, XGBoostMetaLearner):
        return ensemble.fit(members, y, context=context, feature_names=feature_names)
    return ensemble.fit(members, y, context=context)


def predict_with_ensemble(
    ensemble, members: np.ndarray, context: np.ndarray, hours: np.ndarray
) -> np.ndarray:
    """Dispatch to each ensemble's own prediction signature."""
    if isinstance(ensemble, HourlyMOS):
        return ensemble.predict(members, hours=hours)
    return ensemble.predict(members, context=context)


# --------------------------------------------------------------------------- #
def fit_ensembles_for_station(
    slug: str, cfg: Config, *, overwrite: bool = False
) -> Optional[Dict[str, object]]:
    """Fit every configured ensemble for one station."""
    payload, base_meta = load_member_predictions(cfg, slug)
    station = str(base_meta["station_name"])
    member_names: List[str] = list(base_meta["model_names"])
    _assert_fitted_on_validation(payload, base_meta)

    out_dir = cfg.ensemble_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / ENSEMBLE_META).exists() and not overwrite:
        logger.info("%s: ensembles already fitted; use --overwrite to refit.", station)
        with (out_dir / ENSEMBLE_META).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    logger.info("=" * 78)
    logger.info("Ensembles | %s", station)

    members = payload["val_members"].astype(np.float64)
    y = payload["val_y"].astype(np.float64)
    context = payload["val_context"].astype(np.float64)
    hours = _hours_of(payload["val_target_times"])
    check_finite(members, name=f"{station}/val members")

    feature_names = member_names + list(base_meta.get("context_columns", []))
    n, horizon, _ = members.shape
    context_flat = context.reshape(n * horizon, -1)

    info: Dict[str, object] = {
        "station_name": station,
        "slug": slug,
        "member_names": member_names,
        "n_val_windows": n,
        "strategies": {},
        "diagnostics": {},
    }

    for name in cfg.ensembles.names:
        set_global_seed(cfg.seed, deterministic=cfg.deterministic)
        ensemble = build_ensemble(name, cfg, seed=cfg.seed)
        try:
            _fit_one(name, ensemble, members, y, context_flat, hours, member_names, feature_names)
        except (ValueError, RuntimeError, ImportError) as exc:
            logger.error("%s: ensemble %s could not be fitted -- %s", station, name, exc)
            continue

        joblib.dump(ensemble, out_dir / f"{name}.joblib")
        info["strategies"][name] = ENSEMBLE_STRATEGY.get(name, "unknown")

        prediction = predict_with_ensemble(ensemble, members, context_flat, hours)
        stats = continuous_metrics(y, prediction)
        logger.info(
            "%s: %-9s (%-24s) validation RMSE %.3f | MAE %.3f | R %.3f",
            station, name, ENSEMBLE_STRATEGY.get(name, "?"),
            stats["RMSE"], stats["MAE"], stats["R"],
        )
        info["diagnostics"][name] = {
            "val_RMSE": stats["RMSE"], "val_MAE": stats["MAE"], "val_R": stats["R"],
        }

        # Ensemble-specific artefacts worth reporting in their own right.
        if name == "pwe" and getattr(ensemble, "weights_", None) is not None:
            info["diagnostics"][name]["weights"] = {
                m: float(w) for m, w in zip(member_names, ensemble.weights_)
            }
        if name == "sms":
            info["diagnostics"][name].update(
                {
                    "selector_train_accuracy": float(ensemble.train_accuracy_),
                    "oracle_mae_gain_vs_average": float(ensemble.oracle_gain_),
                }
            )
        if name == "h_mos":
            info["diagnostics"][name]["hourly_coefficients"] = ensemble.table().to_dict("records")
        if name == "xgb_meta":
            importances = ensemble.importances()
            if not importances.empty:
                info["diagnostics"][name]["top_features"] = importances.head(10).to_dict("records")

        del ensemble
        gc.collect()

    if not info["strategies"]:
        raise IntegrityError(f"{station}: no ensemble could be fitted.")

    with (out_dir / ENSEMBLE_META).open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)
    return info


# --------------------------------------------------------------------------- #
def run_ensemble_training(cfg: Config, *, overwrite: bool = False) -> List[str]:
    """Fit ensembles for every station that has base models."""
    section(logger, "STAGE 3 / ENSEMBLES  (paper 4.2)")
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)

    slugs = sorted(
        p.name for p in cfg.artifacts_dir.iterdir()
        if p.is_dir() and (p / MEMBER_FILE).exists()
    )
    if not slugs:
        raise FileNotFoundError(f"No stage-2 artefacts in {cfg.artifacts_dir}.")

    fitted: List[str] = []
    for slug in slugs:
        try:
            if fit_ensembles_for_station(slug, cfg, overwrite=overwrite) is not None:
                fitted.append(slug)
        except (IntegrityError, ValueError, KeyError, FileNotFoundError) as exc:
            logger.error("%s: ensemble stage failed -- %s", slug, exc)

    section(logger, f"ENSEMBLE FITTING COMPLETE -- {len(fitted)}/{len(slugs)} stations")
    if not fitted:
        raise IntegrityError("No station produced ensembles.")
    return fitted
