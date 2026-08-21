"""Tests that pin the implementation to the published method.

Each test names the paper section or equation it protects. If a default changes
later, one of these fails and says which claim in the manuscript it just broke.
That is the point: the repository is only useful to a reader if the code and the
paper cannot silently drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pm25forecast.config import ConfigError, load_config
from pm25forecast.data.quality import IQRScreen
from pm25forecast.features import (
    add_cyclical_time_features,
    hour_offset_grid,
    wind_to_components,
)
from pm25forecast.metrics import continuous_metrics, daily_means, split_days
from pm25forecast.models.ensembles import (
    AVAILABLE_ENSEMBLES,
    ENSEMBLE_STRATEGY,
    HourlyMOS,
    PerformanceWeightedEnsemble,
    SimpleAveraging,
    SmartModelSelector,
)


def _cfg(**overrides):
    base = {"paths": {"raw_dir": "."}}
    base.update(overrides)
    return load_config(overrides=base, require_raw=False)


# --------------------------------------------------------------------------- #
# 2.2 -- preprocessing
# --------------------------------------------------------------------------- #
def test_default_splits_match_paper():
    cfg = _cfg()
    assert cfg.split.development == ("2015-01-01 00:00", "2023-12-30 23:00")
    assert cfg.split.test == ("2024-01-01 00:00", "2024-12-30 23:00")
    assert cfg.split.base_train_ratio == pytest.approx(0.70)
    assert cfg.split.gap_hours == 48
    assert cfg.split.meta_train_ratio == pytest.approx(0.80)


def test_test_year_must_follow_development():
    with pytest.raises(ConfigError, match="strictly in the future"):
        _cfg(split={"development": ["2015-01-01 00:00", "2024-12-30 23:00"],
                    "test": ["2024-01-01 00:00", "2024-12-30 23:00"]})


def test_iqr_multiplier_is_one_point_five():
    assert _cfg().outliers.k == pytest.approx(1.5)


def test_iqr_bounds_come_from_training_only():
    """Changing the test distribution must not move the screening bounds."""
    train = pd.DataFrame({"x": np.arange(100.0)})
    screen = IQRScreen(k=1.5).fit(train)
    bounds_before = dict(screen.bounds_)

    # A wildly different "test" frame is transformed, never fitted.
    screen.transform(pd.DataFrame({"x": np.arange(100.0) * 1000}))
    assert screen.bounds_ == bounds_before


def test_iqr_voids_rather_than_clips():
    """An excluded value must become NaN, not the boundary.

    Clipping converts a detectable fault into a plausible observation; applied to
    the target it truncates exactly the pollution episodes the model exists to
    predict.
    """
    train = pd.DataFrame({"x": [10.0, 12.0, 14.0, 16.0, 18.0]})
    screen = IQRScreen(k=1.5).fit(train)
    out, n = screen.transform(pd.DataFrame({"x": [14.0, 500.0]}))
    assert n == 1
    assert np.isnan(out["x"].iloc[1]), "The outlier was clipped instead of voided."
    assert out["x"].iloc[0] == 14.0


def test_iqr_screen_refuses_transform_before_fit():
    with pytest.raises(RuntimeError):
        IQRScreen().transform(pd.DataFrame({"x": [1.0]}))


def test_imputer_and_scaler_must_be_train_fitted():
    with pytest.raises(ConfigError, match="fitted on training data only"):
        _cfg(imputation={"fit_on": "development"})
    with pytest.raises(ConfigError, match="exclusively from the training set"):
        _cfg(scaling={"fit_on": "development"})
    with pytest.raises(ConfigError, match="leakage"):
        _cfg(outliers={"fit_on": "all"})


# --------------------------------------------------------------------------- #
# 2.2 -- features (Eqs. 1-4)
# --------------------------------------------------------------------------- #
def test_wind_components_follow_equations_1_and_2():
    index = pd.date_range("2024-01-01", periods=4, freq="h")
    frame = pd.DataFrame(
        {"wind_speed": [10.0] * 4, "wind_direction": [0.0, 90.0, 180.0, 270.0]}, index=index
    )
    out = wind_to_components(frame)
    assert out["U"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert out["V"].iloc[0] == pytest.approx(-10.0)
    assert out["U"].iloc[1] == pytest.approx(-10.0)
    assert "wind_speed" not in out.columns and "wind_direction" not in out.columns


def test_cyclical_hour_features_are_circular():
    """Eqs. 3-4: 23:00 and 00:00 must be neighbours on the unit circle."""
    index = pd.DatetimeIndex(["2024-01-01 23:00", "2024-01-02 00:00"])
    out = add_cyclical_time_features(pd.DataFrame(index=index), ["hour"])
    assert {"Hour_sin", "Hour_cos"} <= set(out.columns)
    distance = np.hypot(
        out["Hour_sin"].iloc[1] - out["Hour_sin"].iloc[0],
        out["Hour_cos"].iloc[1] - out["Hour_cos"].iloc[0],
    )
    assert distance < 0.3


def test_hour_offset_spans_the_whole_horizon():
    """Hour_offset must run 0..47 for a 48-hour forecast (paper §2.2)."""
    grid = hour_offset_grid(3, 48)
    assert grid.shape == (3, 48)
    assert grid.min() == 0 and grid.max() == 47
    assert (grid[0] == np.arange(48)).all()


def test_window_geometry_matches_table_3():
    cfg = _cfg()
    assert cfg.windows.input_hours == 48
    assert cfg.windows.horizon_hours == 48
    assert cfg.features.hour_offset_feature is True


# --------------------------------------------------------------------------- #
# 4.1, Table 2 -- base learners
# --------------------------------------------------------------------------- #
def test_four_base_learners_with_paper_hyperparameters():
    cfg = _cfg().base_models
    assert cfg.names == ["lstm", "gru", "bi_lstm", "bi_gru"]
    assert tuple(cfg.rnn_units) == (64, 32)      # Table 2, "RNNs"
    assert tuple(cfg.bi_units) == (32, 32)       # Table 2, "Bi-RNNs"
    assert cfg.epochs == 100
    assert cfg.batch_size == 32
    assert cfg.early_stopping_patience == 10
    assert cfg.learning_rate == pytest.approx(1e-3)


def test_loss_assignment_matches_table_2():
    """Table 2: RNNs use IWL, Bi-RNNs use EEL."""
    from pm25forecast.models.losses import DEFAULT_LOSS_BY_MODEL

    assert DEFAULT_LOSS_BY_MODEL["lstm"] == "iwl"
    assert DEFAULT_LOSS_BY_MODEL["gru"] == "iwl"
    assert DEFAULT_LOSS_BY_MODEL["bi_lstm"] == "eel"
    assert DEFAULT_LOSS_BY_MODEL["bi_gru"] == "eel"


def test_iwl_weights_low_concentrations_more(tf_losses):
    """Eq. 19: the weight 1 + alpha/(y+1) falls as y grows."""
    loss, tf = tf_losses["iwl"], tf_losses["tf"]
    low = float(loss(tf.constant([[0.0]]), tf.constant([[0.1]])))
    high = float(loss(tf.constant([[1.0]]), tf.constant([[1.1]])))
    assert low > high
    # Exact ratio: (1 + 10/1) / (1 + 10/2) = 11 / 6
    assert low / high == pytest.approx(11 / 6, rel=1e-4)


def test_eel_weights_high_concentrations_exponentially(tf_losses):
    """Eq. 20: the weight exp(y) grows exponentially with y."""
    loss, tf = tf_losses["eel"], tf_losses["tf"]
    high = float(loss(tf.constant([[1.0]]), tf.constant([[0.9]])))
    low = float(loss(tf.constant([[0.0]]), tf.constant([[0.1]])))
    assert high / low == pytest.approx(np.e, rel=1e-3)


def test_eel_refuses_unscaled_targets():
    """exp(300) overflows; the failure must be an error, not silent NaN weights."""
    from pm25forecast.models.losses import resolve_loss

    with pytest.raises(ValueError, match="scaled"):
        resolve_loss("eel", targets_are_scaled=False)


def test_losses_ignore_missing_labels(tf_losses):
    tf = tf_losses["tf"]
    for key in ("iwl", "eel"):
        loss = tf_losses[key]
        y = tf.constant([[0.2, float("nan"), 0.8]])
        assert float(loss(y, tf.constant([[0.2, 99.0, 0.8]]))) < 1e-5


# --------------------------------------------------------------------------- #
# 4.2, Table 2 -- the six ensembles
# --------------------------------------------------------------------------- #
def test_six_ensembles_in_three_strategies():
    cfg = _cfg().ensembles
    assert cfg.names == ["sa", "pwe", "mlp_meta", "xgb_meta", "h_mos", "sms"]
    assert len(set(ENSEMBLE_STRATEGY.values())) == 3
    assert set(AVAILABLE_ENSEMBLES) == set(cfg.names)


def test_ensemble_hyperparameters_match_table_2():
    cfg = _cfg().ensembles
    assert tuple(cfg.mlp_hidden) == (32, 16, 8)
    assert (cfg.xgb_n_estimators, cfg.xgb_max_depth) == (100, 6)
    assert cfg.xgb_learning_rate == pytest.approx(0.1)
    assert (cfg.sms_n_estimators, cfg.sms_max_depth) == (100, 6)
    assert cfg.pwe_method == "SLSQP"


def test_simple_averaging_is_the_uniform_mean():
    """Eq. 6."""
    members = np.array([[[10.0, 20.0, 30.0, 40.0]]])
    assert SimpleAveraging().predict(members)[0, 0] == pytest.approx(25.0)


def test_pwe_weights_are_convex_and_recover_the_truth():
    """Eqs. 7-8: weights sum to 1, lie in [0, 1], and favour the accurate member."""
    rng = np.random.default_rng(0)
    truth = rng.uniform(10, 80, (200, 48))
    good = truth + rng.normal(0, 1, truth.shape)
    bad = truth + rng.normal(0, 20, truth.shape)
    members = np.stack([good, bad], axis=-1)

    pwe = PerformanceWeightedEnsemble().fit(members, truth)
    assert pwe.weights_.sum() == pytest.approx(1.0)
    assert (pwe.weights_ >= -1e-9).all() and (pwe.weights_ <= 1 + 1e-9).all()
    assert pwe.weights_[0] > pwe.weights_[1], "The accurate member must get more weight."

    prediction = pwe.predict(members)
    assert prediction.shape == truth.shape
    # A convex combination can never leave the members' envelope.
    assert (prediction >= members.min(axis=2) - 1e-9).all()
    assert (prediction <= members.max(axis=2) + 1e-9).all()


def test_pwe_beats_simple_averaging_when_members_differ_in_skill():
    rng = np.random.default_rng(1)
    truth = rng.uniform(10, 80, (300, 48))
    members = np.stack(
        [truth + rng.normal(0, 1, truth.shape), truth + rng.normal(0, 25, truth.shape)], axis=-1
    )
    sa_rmse = continuous_metrics(truth, SimpleAveraging().predict(members))["RMSE"]
    pwe_rmse = continuous_metrics(
        truth, PerformanceWeightedEnsemble().fit(members, truth).predict(members)
    )["RMSE"]
    assert pwe_rmse < sa_rmse


def test_h_mos_learns_an_hour_specific_correction():
    """Eq. 11: a bias that exists only at certain hours must be removed there."""
    rng = np.random.default_rng(2)
    n, horizon = 200, 48
    hours = np.tile(np.arange(horizon) % 24, (n, 1))
    truth = 30 + rng.normal(0, 3, (n, horizon))
    biased = truth.copy()
    biased[hours == 6] -= 20.0            # a large bias, only at 06:00
    members = np.stack([biased, biased], axis=-1)

    mos = HourlyMOS().fit(members, truth, hours=hours)
    corrected = mos.predict(members, hours=hours)

    before = np.abs(biased[hours == 6] - truth[hours == 6]).mean()
    after = np.abs(corrected[hours == 6] - truth[hours == 6]).mean()
    assert after < 0.25 * before, "H-MOS failed to remove an hour-specific bias."

    table = mos.table()
    assert len(table) == 24 and set(table.columns) == {"hour", "alpha", "beta"}


def test_sms_picks_the_locally_better_member():
    """Eqs. 12-13: with a context that determines which member is right, the
    selector must learn it."""
    rng = np.random.default_rng(3)
    n, horizon = 300, 48
    truth = rng.uniform(10, 80, (n, horizon))
    regime = (rng.random((n, horizon)) > 0.5).astype(float)

    a = np.where(regime == 1, truth, truth + 30)
    b = np.where(regime == 0, truth, truth + 30)
    members = np.stack([a, b], axis=-1)
    context = regime.reshape(-1, 1)

    sms = SmartModelSelector(n_estimators=30, max_depth=4).fit(
        members, truth, context=context, member_names=["a", "b"]
    )
    prediction = sms.predict(members, context=context)
    assert continuous_metrics(truth, prediction)["RMSE"] < continuous_metrics(
        truth, members.mean(axis=-1)
    )["RMSE"]
    assert sms.train_accuracy_ > 0.9


# --------------------------------------------------------------------------- #
# 4.4, Eqs. 14-18 -- metrics
# --------------------------------------------------------------------------- #
def test_daily_means_split_the_horizon_into_two_days():
    values = np.concatenate([np.full(24, 10.0), np.full(24, 50.0)]).reshape(1, 48)
    means = daily_means(values)
    assert means.shape == (1, 2)
    assert means[0, 0] == pytest.approx(10.0)
    assert means[0, 1] == pytest.approx(50.0)
    days = split_days(values)
    assert set(days) == {"day1", "day2"}


def test_daily_mean_requires_completeness():
    values = np.full((1, 48), 20.0)
    values[0, :20] = np.nan                    # only 4 valid hours on day 1
    means = daily_means(values, min_hours=18)
    assert np.isnan(means[0, 0])
    assert means[0, 1] == pytest.approx(20.0)


def test_pearson_r_and_ioa_disagree_on_a_biased_forecast():
    """Eq. 16 vs Eq. 17 -- reporting both is what distinguishes 'wrong shape'
    from 'right shape, wrong level'."""
    truth = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    biased = truth + 30.0
    stats = continuous_metrics(truth, biased)
    assert stats["R"] == pytest.approx(1.0)
    assert stats["IOA"] < 0.6
    assert stats["MBE"] == pytest.approx(30.0)


def test_perfect_forecast_scores_perfectly():
    y = np.array([12.0, 25.0, 48.0, 7.0])
    stats = continuous_metrics(y, y)
    assert stats["RMSE"] == 0.0 and stats["MAE"] == 0.0
    assert stats["R"] == pytest.approx(1.0)
    assert stats["IOA"] == pytest.approx(1.0)
    assert stats["Unc_percent"] == pytest.approx(0.0)


def test_uncertainty_is_zero_for_a_constant_error():
    """Eq. 18 measures the spread of |error| about the MAE, so a perfectly
    consistent bias has zero uncertainty despite a large MAE."""
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    stats = continuous_metrics(truth, truth + 5.0)
    assert stats["MAE"] == pytest.approx(5.0)
    assert stats["Unc_percent"] == pytest.approx(0.0, abs=1e-9)


def test_metrics_mask_missing_values():
    truth = np.array([1.0, np.nan, 3.0, 4.0])
    pred = np.array([1.0, 2.0, np.nan, 4.0])
    assert continuous_metrics(truth, pred)["n"] == 2


# --------------------------------------------------------------------------- #
# Guards on things the paper does NOT do
# --------------------------------------------------------------------------- #
def test_extras_are_off_by_default():
    cfg = _cfg()
    assert cfg.evaluation.baselines == []
    assert cfg.features.add_day_of_year is False
    assert cfg.imputation.short_gap_hours == 0
    assert cfg.evaluation.bootstrap_samples == 0


def test_overlapping_test_windows_are_rejected():
    with pytest.raises(ConfigError, match="independent samples"):
        _cfg(windows={"horizon_hours": 48, "stride": {"test": 6}})


def test_gap_smaller_than_horizon_warns():
    with pytest.warns(RuntimeWarning, match="overlap in time"):
        _cfg(split={"gap_hours": 4})


def test_unknown_config_key_is_an_error():
    with pytest.raises(ConfigError, match="Unknown configuration key"):
        _cfg(ensembless={"names": []})


@pytest.fixture(scope="module")
def tf_losses():
    """Instantiated custom losses, imported lazily so pure-config tests stay fast."""
    import tensorflow as tf

    from pm25forecast.models.losses import resolve_loss

    return {"tf": tf, "iwl": resolve_loss("iwl", alpha=10.0), "eel": resolve_loss("eel", power=2.0)}


# --------------------------------------------------------------------------- #
# Fair comparison -- physical bounds apply to every forecaster
# --------------------------------------------------------------------------- #
def test_prediction_clip_is_applied_to_every_forecaster_or_to_none():
    """The physical bound must not be applied to the ensemble alone.

    A negative PM2.5 concentration is impossible for every model equally, so the
    clip is a property of the quantity. Applying it only to the ensembles would shrink
    the ensembles' error without shrinking the members' -- an advantage of the same order
    as the margin the paper reports, manufactured by the scoring code rather than
    earned by the method.
    """
    import inspect

    from pm25forecast.pipeline import evaluate as evaluate_module

    source = inspect.getsource(evaluate_module.evaluate_station)
    clip_line = "name: np.clip(prediction, clip[0], clip[1])"
    assert clip_line in source, (
        "the uniform clip over every entry of `forecasts` is gone -- "
        "check that the bound is not being applied to a subset of models"
    )
    # ...and that it happens after every forecaster is in the dict, i.e. before scoring.
    assert source.index(clip_line) < source.index("for name, prediction in forecasts.items():")
