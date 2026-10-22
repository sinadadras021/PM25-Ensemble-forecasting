"""One-epoch integration test: actual networks, synthetic data, no paper results."""

from dataclasses import replace
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from pm25forecast.analysis import analyze
from pm25forecast.config import BASE_NAMES, ENSEMBLE_NAMES, sha256
from pm25forecast.pipeline import (
    load_prepared,
    predict,
    prepare,
    require_training,
    train,
)
from pm25forecast.windows import WindowPool, meta_partition


@pytest.fixture(scope="module")
def trained(synthetic_config):
    train(synthetic_config)
    return synthetic_config


def test_actual_training_reload_prediction_and_analysis(trained):
    cfg = trained
    ready = require_training(cfg)
    assert ready["base_models"] == 4 and ready["ensemble_models"] == 6
    assert ready["scope"] == "pooled"
    first = predict(cfg)
    second = predict(cfg)
    np.testing.assert_allclose(
        first[BASE_NAMES + ENSEMBLE_NAMES],
        second[BASE_NAMES + ENSEMBLE_NAMES],
        rtol=0,
        atol=1e-5,
    )
    assert len(first) == 2 * 4 * 48
    assert first.actual.isna().sum() == 2
    np.testing.assert_allclose(first.sa, first[BASE_NAMES].mean(axis=1), atol=1e-6)
    state = joblib.load(cfg.out / "ensembles" / "ensembles.joblib")
    assert state["pwe"].sum() == pytest.approx(1, abs=1e-6)
    assert np.min(state["pwe"]) >= -1e-10
    np.testing.assert_allclose(
        first.pwe, first[BASE_NAMES].to_numpy() @ state["pwe"], atol=1e-5
    )
    assert (
        np.isclose(first[BASE_NAMES].to_numpy(), first.sms.to_numpy()[:, None])
        .any(axis=1)
        .all()
    )
    table = analyze(cfg)
    assert len(table) == 20 and table.stations.eq(2).all()
    assert len(pd.read_csv(cfg.out / "DM_Holm.csv")) == 8
    assert len(pd.read_csv(cfg.out / "city_warning_metrics.csv")) == 20


def test_resume_preserves_all_model_files(trained):
    paths = list((trained.out / "models").glob("*.h5")) + list(
        (trained.out / "ensembles").glob("*.h5")
    )
    before = {path.name: sha256(path) for path in paths}
    train(trained)
    assert {path.name: sha256(path) for path in paths} == before


def test_windows_temporal_meta_split_and_issue_row(trained):
    obj = load_prepared(trained)
    pool = WindowPool(obj, trained, "val")
    fit, valid = meta_partition(pool, trained.meta_train_ratio)
    assert pool.first_targets[fit].max() < pool.first_targets[valid].min()
    assert (
        pool.exact_indices().context_last_observation.max()
        <= pool.exact_indices().issue_time.max()
    )
    test = WindowPool(obj, trained, "test")
    before_x, _ = test.sample(0)
    before_context = test.context()[0].copy()
    _, station, _, target = test.records[0]
    item = obj["stations"][test.names[int(station)]]
    # Alter every future channel, including weather and verification labels.
    item["physical"][target:] = 1e6
    item["x"][target:] = 1e6
    item["y"][target:] = np.nan
    after_x, _ = test.sample(0)
    np.testing.assert_array_equal(before_x, after_x)
    np.testing.assert_array_equal(before_context, test.context()[0])
    assert len(WindowPool(obj, trained, "test")) == len(test)
    # The final row uses unscreened observations, never a centered future QC flag.
    item["x"][target - 1] = -999
    patched_x, _ = test.sample(0)
    np.testing.assert_array_equal(patched_x[-1], item["last_x"][target - 1])


def test_configuration_and_artifact_integrity_guards(trained):
    with pytest.raises(ValueError, match="different"):
        load_prepared(replace(trained, seed=43))
    path = trained.out / "models" / "gru.h5"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"corruption-check")
        with pytest.raises(ValueError, match="Invalid"):
            require_training(trained)
    finally:
        path.write_bytes(original)


def test_both_real_shap_explainers_on_tiny_sample(trained):
    from pm25forecast.explain import explain

    explain(
        trained,
        base_background_size=2,
        base_sample_size=2,
        gradient_samples=4,
        meta_background_size=2,
        meta_sample_size=2,
        kernel_samples=32,
    )
    folder = trained.out / "SHAP"
    for day in [1, 2]:
        data = np.load(folder / f"BiGRU_Day{day}.npz")
        assert data["lag_values"].shape == (2, 48, 14)
        assert np.isfinite(data["lag_values"]).all()
    meta = np.load(folder / "MLP.npz")
    assert meta["shap"].shape == (2, 13)
    assert np.max(np.abs(meta["residual"])) <= 0.001


def test_masked_mse_and_architecture_contract():
    import tensorflow as tf
    from pm25forecast.models import build_base, build_meta, callbacks, observed_mse
    from pm25forecast.config import Config

    loss = observed_mse()
    assert (
        float(
            loss(
                tf.constant([[[1.0], [float("nan")]]]), tf.constant([[[3.0], [999.0]]])
            )
        )
        == 4
    )
    cfg = Config()
    for name in BASE_NAMES:
        model = build_base(name, 14, cfg)
        assert model.output_shape == (None, 48, 1)
        assert model.layers[-2].layer.units == 32
        assert model.layers[-1].layer.units == 1
    assert [layer.units for layer in build_meta(13, cfg).layers] == [32, 16, 8, 1]
    assert [type(callback).__name__ for callback in callbacks(cfg)] == [
        "EarlyStopping",
        "ReduceLROnPlateau",
        "TerminateOnNaN",
    ]


def test_three_separate_plot_exports(trained, monkeypatch):
    """Exercise actual plotting, using low raster DPI only for this smoke test."""
    from matplotlib.figure import Figure
    from pm25forecast.plots import plot_shap

    original_save = Figure.savefig
    exports = []

    def small_save(figure, filename, **kwargs):
        kwargs["dpi"] = 36
        original_save(figure, filename, **kwargs)
        exports.append(filename)

    monkeypatch.setattr(Figure, "savefig", small_save)
    plot_shap(trained)
    assert len(exports) == 9
    assert len({path.stem for path in exports}) == 3
    assert all(path.exists() and path.stat().st_size > 0 for path in exports)
