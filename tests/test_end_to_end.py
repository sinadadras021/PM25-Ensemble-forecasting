"""End-to-end smoke run.

Trains real (tiny) networks on synthetic stations and walks every stage. This is
the test that proves the pipeline has no plumbing bugs: shapes line up, artefacts
are written and reloaded, the ensembles refuse test-fitted inputs, and the result
tables come out populated across all three strategies.

Marked ``slow`` -- run with ``pytest -m slow`` or in CI, not on every save.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow


def test_full_pipeline(smoke_config):
    from pm25forecast.pipeline.evaluate import run_evaluation
    from pm25forecast.pipeline.preprocess import run_preprocessing
    from pm25forecast.pipeline.train_base import run_base_training
    from pm25forecast.pipeline.train_ensembles import run_ensemble_training

    cfg = smoke_config

    # -- stage 1 ------------------------------------------------------------- #
    quality = run_preprocessing(cfg)
    assert {"station", "accepted", "reason"} <= set(quality.columns)
    assert quality["accepted"].all(), quality[["station", "reason"]].to_dict("records")

    slug = quality["slug"].iloc[0]
    meta = json.loads(
        (cfg.cleaned_dir / f"{slug}_meta.json").read_text(encoding="utf-8")
    )
    assert meta["outliers_removed_train"] > 0, "The injected spikes should have been screened."
    assert meta["target_input_column"] in meta["feature_columns"], (
        "The gap-filled PM2.5 history must be an encoder input; it is available at issue "
        "time and is the strongest single predictor."
    )
    assert cfg.data.target not in meta["feature_columns"], (
        "The raw label column must never be an encoder input."
    )
    for dropped in cfg.data.drop_columns:
        assert dropped not in meta["feature_columns"]

    # -- stage 2 ------------------------------------------------------------- #
    trained = run_base_training(cfg)
    assert trained
    slug = trained[0]
    for name in cfg.base_models.names:
        assert (cfg.artifacts_dir / slug / f"{name}.keras").exists()

    payload = json.loads(
        (cfg.artifacts_dir / slug / "base_model_meta.json").read_text(encoding="utf-8")
    )
    assert set(payload["validation_metrics"]) == set(cfg.base_models.names)
    assert all(m["RMSE"] > 0 for m in payload["validation_metrics"].values()), (
        "A zero validation RMSE means the labels leaked into the inputs."
    )

    # -- stage 3 ------------------------------------------------------------- #
    fitted = run_ensemble_training(cfg)
    assert fitted
    info = json.loads(
        (cfg.ensemble_dir / slug / "ensemble_meta.json").read_text(encoding="utf-8")
    )
    assert set(info["strategies"]) == set(cfg.ensembles.names), (
        "Every configured ensemble must be fitted, or the strategy comparison is incomplete."
    )
    assert len(set(info["strategies"].values())) == 3, "All three strategies must be represented."
    if "pwe" in info["diagnostics"] and "weights" in info["diagnostics"]["pwe"]:
        assert sum(info["diagnostics"]["pwe"]["weights"].values()) == pytest.approx(1.0)
    if "h_mos" in info["diagnostics"]:
        assert len(info["diagnostics"]["h_mos"]["hourly_coefficients"]) == 24

    # -- stage 4 ------------------------------------------------------------- #
    results = run_evaluation(cfg)
    assert not results.empty
    models = set(results["model"])
    assert set(cfg.base_models.names) <= models
    assert set(cfg.ensembles.names) <= models

    for column in ("day1_RMSE", "day2_RMSE", "day1_R", "day1_IOA", "day1_Unc_percent"):
        assert column in results.columns, f"paper 4.4 requires {column}"

    finite = results["day1_RMSE"].dropna()
    assert (finite > 0).all(), "Zero held-out RMSE means the labels leaked."
    assert (finite < 500).all(), "Implausible errors indicate a scaling bug."

    for name in ("metrics_all_stations.csv", "metrics_city_mean.csv", "metrics_by_strategy.csv"):
        assert (cfg.results_dir / name).exists()

    strategies = pd.read_csv(cfg.results_dir / "metrics_by_strategy.csv")
    assert len(strategies) >= 3

    # Each window must cover exactly two calendar days.
    predictions = pd.read_parquet(cfg.results_dir / f"predictions_{slug}.parquet")
    first_hours = predictions.loc[predictions["lead_hour"] == 1, "timestamp"].dt.hour
    assert first_hours.nunique() == 1 and first_hours.iloc[0] == 0
    assert set(predictions["forecast_day"].unique()) == {1, 2}


def test_simple_averaging_is_reproduced_exactly(smoke_config):
    """SA has no parameters, so the saved prediction must equal the plain mean
    of the saved member columns -- a direct check that nothing is mislabelled."""
    from pm25forecast.pipeline.evaluate import run_evaluation
    from pm25forecast.pipeline.preprocess import run_preprocessing
    from pm25forecast.pipeline.train_base import run_base_training
    from pm25forecast.pipeline.train_ensembles import run_ensemble_training

    cfg = smoke_config
    run_preprocessing(cfg)
    slug = run_base_training(cfg)[0]
    run_ensemble_training(cfg)
    run_evaluation(cfg)

    frame = pd.read_parquet(cfg.results_dir / f"predictions_{slug}.parquet")
    members = frame[list(cfg.base_models.names)].to_numpy()
    np.testing.assert_allclose(frame["sa"].to_numpy(), members.mean(axis=1), rtol=1e-6)


def test_ensembles_refuse_non_chronological_blocks(smoke_config):
    """The guard that keeps the stack honest must actually fire."""
    from pm25forecast.pipeline.preprocess import run_preprocessing
    from pm25forecast.pipeline.train_base import run_base_training
    from pm25forecast.pipeline.train_ensembles import (
        _assert_fitted_on_validation,
        load_member_predictions,
    )
    from pm25forecast.validation import IntegrityError

    cfg = smoke_config
    run_preprocessing(cfg)
    slug = run_base_training(cfg)[0]
    payload, meta = load_member_predictions(cfg, slug)

    _assert_fitted_on_validation(payload, meta)          # correct data passes

    tampered = dict(payload)
    tampered["val_members"] = payload["test_members"]     # substituting test data
    with pytest.raises(IntegrityError):
        _assert_fitted_on_validation(tampered, meta)
