"""Numerical and protocol checks independent of neural-network fitting."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pm25forecast.analysis import dm_test, holm, metrics, warning
from pm25forecast.config import BASE_NAMES, Config, METEOROLOGY
from pm25forecast.data import (
    convert_weather,
    four_condition_screen,
    load_network,
    read_frame,
)
from pm25forecast.ensembles import flat_design


def test_paper_defaults():
    cfg = Config()
    assert len(cfg.features) == 14 and "dwpc" not in cfg.features
    assert len(BASE_NAMES + cfg.context) == 13
    assert (cfg.batch_size, cfg.epochs, cfg.seed) == (512, 100, 42)
    assert (cfg.early_stopping_patience, cfg.reduce_lr_patience) == (10, 5)


@pytest.mark.parametrize(
    "changes",
    [
        {"gap_hours": 47},
        {"validation_stride": 24},
        {"test_stride": 24},
        {"input_hours": 24},
        {"dropout": 1},
        {"knn_neighbors": 0},
        {"exclude_groups": ["unknown"]},
        {"merged_meteorology_alignment_verified": False},
    ],
)
def test_invalid_protocol_is_rejected(synthetic_config, changes):
    with pytest.raises(ValueError):
        replace(synthetic_config, **changes).validate()


def test_weather_unit_conversion(synthetic_config):
    frame = pd.DataFrame(
        {
            "tmpc": [20],
            "relh": [50],
            "alti": [30],
            "vsby": [10],
            "wind_speed": [10],
            "wind_direction": [90],
        }
    )
    cfg = replace(
        synthetic_config,
        visibility_unit="mile",
        pressure_unit="inHg",
        wind_unit="knots",
    )
    converted = convert_weather(frame, cfg)
    np.testing.assert_allclose(converted.U, [-5.144444444444445])
    np.testing.assert_allclose(converted.V, [0], atol=1e-12)
    np.testing.assert_allclose(converted.vsby, [16.09344])
    np.testing.assert_allclose(converted.alti, [1015.916600001])


def test_timezone_and_duplicate_guard(tmp_path, synthetic_config):
    path = tmp_path / "clock.csv"
    pd.DataFrame({"Date": ["2024-01-01 00:00:00"], "x": [1]}).to_csv(path, index=False)
    frame = read_frame(path, synthetic_config, "Asia/Tehran")
    assert frame.index[0] == pd.Timestamp("2023-12-31 20:30:00Z")
    pd.DataFrame({"Date": ["2024-01-01 00:00:00"] * 2}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        read_frame(path, synthetic_config, "UTC")


def test_separate_weather_never_joins_future_report(tmp_path, synthetic_config):
    met = tmp_path / "weather.csv"
    data = {"Date": ["2023-10-01 00:00:00", "2023-10-01 01:00:00"]}
    data.update({name: [10, 999] for name in METEOROLOGY})
    pd.DataFrame(data).to_csv(met, index=False)
    cfg = replace(
        synthetic_config,
        meteorology_file=str(met),
        meteorology_reporting_delay_minutes=30,
    )
    frames, _ = load_network(cfg)
    weather = next(iter(frames.values()))
    assert np.isnan(weather.iloc[0].tmpc)
    assert weather.iloc[1].tmpc == 10  # 01:00 report only becomes available at 01:30.


def test_individual_meta_inputs_and_nonfinite_guard():
    members = np.broadcast_to(np.arange(4), (2, 48, 4)).copy()
    context = np.ones((2, 48, 9))
    design = flat_design(members, context)
    assert design.shape == (96, 13)
    np.testing.assert_array_equal(design[:, :4], np.tile(np.arange(4), (96, 1)))
    context[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        flat_design(members, context)


def test_warning_counts_threshold_and_standard_scores():
    result = warning([40, 40, 20, 35, np.nan], [40, 20, 40, 35, 40])
    assert [result[name] for name in ["TP", "FN", "FP", "TN"]] == [1, 1, 1, 1]
    for name in ["POD", "Precision", "FAR", "FPR", "Specificity", "F1", "Accuracy"]:
        assert result[name] == 0.5
    assert result["CSI"] == pytest.approx(1 / 3)
    assert result["HSS"] == 0


def test_metrics_and_holm():
    result = metrics([1, 2, 3, np.nan], [2, 2, 2, 100])
    assert result["RMSE"] == pytest.approx(np.sqrt(2 / 3))
    assert result["MAE"] == pytest.approx(2 / 3)
    corrected = holm(pd.DataFrame({"p_value": [0.04, 0.001, 0.02]}))
    np.testing.assert_allclose(corrected.p_Holm, [0.04, 0.003, 0.04])
    assert np.isnan(dm_test([0, 0])["DM"])
    assert np.isnan(dm_test(np.zeros(10))["DM"])
    assert 0 <= dm_test(np.sin(np.arange(20)) + 0.1)["p_value"] <= 1


def test_four_condition_qc_calibrates_only_on_training(synthetic_config):
    frames, _ = load_network(synthetic_config)
    frames["station_2"] = frames["station_1"].copy()
    cfg = replace(synthetic_config, expected_stations=3)
    _, baseline_calibration, _ = four_condition_screen(frames, cfg)
    spike_time = cfg.timestamp("2024-01-02 12:00:00")
    frames["station_0"].loc[spike_time, "PM 2.5 ug/m3"] = 10000
    flags, calibration, counts = four_condition_screen(frames, cfg)
    assert calibration == baseline_calibration
    assert flags["station_0"].loc[spike_time, "PM 2.5 ug/m3"]
    assert counts.flagged.sum() >= 1
    # A synchronized city episode must fail the station/city excess condition.
    for frame in frames.values():
        frame.loc[spike_time, "PM 2.5 ug/m3"] = 10000
    flags, _, _ = four_condition_screen(frames, cfg)
    assert not any(flag.loc[spike_time, "PM 2.5 ug/m3"] for flag in flags.values())
