"""Small synthetic fixtures; no private data or publication results are used."""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from pm25forecast.config import Config, METEOROLOGY, POLLUTANTS


@pytest.fixture(scope="session")
def synthetic_config(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_pm25")
    raw = root / "raw"
    raw.mkdir()
    index = pd.date_range("2023-10-01", "2024-01-08 23:00:00", freq="h")
    hour = np.arange(len(index), dtype=float)
    for station in range(2):
        data = {"Date": index.strftime("%Y-%m-%d %H:%M:%S")}
        for channel, name in enumerate(POLLUTANTS):
            data[name] = (
                20
                + 3 * station
                + channel
                + 7 * np.sin(hour / (9 + channel))
                + 0.7 * np.cos(hour / 73)
                + hour * 0.0003
            )
        for channel, name in enumerate(METEOROLOGY):
            data[name] = 10 + channel + 2 * np.cos(hour / (13 + channel))
        data["vsby"] = data["vsby"] * 1000
        data["alti"] = data["alti"] + 980
        frame = pd.DataFrame(data)
        # Missing labels must remain missing; missing inputs must be imputed.
        frame.loc[100, POLLUTANTS[0]] = np.nan
        frame.loc[200, "relh"] = np.nan
        frame.loc[len(frame) - 30, POLLUTANTS[0]] = np.nan
        frame.to_csv(raw / f"station_{station}.csv", index=False)
    return Config(
        raw_dir=str(raw),
        output_dir=str(root / "artifacts"),
        merged_meteorology_alignment_verified=True,
        pollutant_timezone="UTC",
        expected_stations=2,
        development_start="2023-10-01 00:00:00",
        development_end="2023-12-31 23:59:59",
        test_start="2024-01-01 00:00:00",
        test_end="2024-01-08 23:59:59",
        train_stride=192,
        epochs=1,
        batch_size=64,
        xgb_trees=3,
        sms_trees=3,
    )
