"""Shared fixtures.

The synthetic generator builds stations from the mechanisms that actually drive
urban PM2.5 -- boundary-layer collapse, traffic emission peaks, wind ventilation,
winter inversions -- and then damages them the way real archives are damaged:
outages of several lengths, instrument spikes, duplicated timestamps, and one
station too sparse to admit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def synthetic_station(
    start: str = "2021-01-01",
    end: str = "2022-12-31 23:00",
    *,
    seed: int = 0,
    target_completeness: float = 0.95,
    inject_faults: bool = True,
) -> pd.DataFrame:
    """One physically plausible hourly PM2.5 station."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq="h")
    n = len(index)
    hour = index.hour.to_numpy()
    doy = index.dayofyear.to_numpy()

    temperature = (
        18.0
        + 14.0 * np.sin(2 * np.pi * (doy - 100) / 365.25)
        + 7.0 * np.sin(2 * np.pi * (hour - 9) / 24)
        + rng.normal(0, 2.0, n)
    )
    humidity = np.clip(60 - 1.2 * (temperature - 18) + rng.normal(0, 8, n), 3, 100)
    u_wind, v_wind = rng.normal(0, 2.5, n), rng.normal(0, 2.5, n)
    ventilation = np.hypot(u_wind, v_wind)
    pressure = 1015 + 6 * np.sin(2 * np.pi * doy / 365.25) + rng.normal(0, 3, n)

    # Boundary-layer height: shallow at night and in winter, deep on summer
    # afternoons. This is the dominant control on surface PM2.5.
    mixing = 0.35 + 0.65 * np.clip(np.sin(np.pi * (hour - 6) / 14), 0, None)
    mixing *= 0.6 + 0.4 * np.clip((temperature + 5) / 35, 0, 1)

    traffic = (
        1.0
        + 0.8 * np.exp(-((hour - 8) ** 2) / 5)
        + 0.9 * np.exp(-((hour - 19) ** 2) / 5)
    )
    no = np.clip(35 * traffic - 3 * ventilation + rng.normal(0, 8, n), 0.5, None)
    no2 = np.clip(0.55 * no + rng.normal(0, 6, n), 0.5, None)
    co = np.clip(0.02 * no + 0.6 + rng.normal(0, 0.2, n), 0.05, None)
    so2 = np.clip(9 + rng.normal(0, 3.0, n), 0.2, None)
    ozone = np.clip(
        20 + 25 * np.clip(np.sin(np.pi * (hour - 6) / 13), 0, None) - 0.15 * no
        + rng.normal(0, 5, n), 0.5, None,
    )

    innovation = rng.normal(0, 4.0, n)
    residual = np.empty(n)
    residual[0] = innovation[0]
    for i in range(1, n):
        residual[i] = 0.85 * residual[i - 1] + innovation[i]

    pm25 = np.clip(
        (14.0 * traffic + 0.10 * no + 6.0) / mixing
        - 1.6 * ventilation
        + residual,
        1.0, None,
    )
    pm10 = pm25 * (2.0 + 0.2 * rng.normal(0, 1, n))

    frame = pd.DataFrame(
        {
            "PM 2.5 ug/m3": pm25, "PM 10 ug/m3": pm10,
            "O3 ppb": ozone, "CO ppm": co, "NO ppb": no, "NO2 ppb": no2,
            "NOx ppb": no + no2, "SO2 ppb": so2,
            "tmpc": temperature, "dwpc": temperature - 8.0, "relh": humidity,
            "alti": pressure, "vsby": np.clip(16 - 0.09 * pm25 + rng.normal(0, 1.5, n), 0.2, 40),
            "feel": temperature - 1.0, "U": u_wind, "V": v_wind,
        },
        index=index,
    )
    frame.index.name = "Date"
    if not inject_faults:
        return frame

    for column, gaps in (
        ("PM 2.5 ug/m3", [(1200, 2), (5000, 8), (9000, 30)]),
        ("NO ppb", [(2000, 1), (7000, 11)]),
        ("tmpc", [(3000, 3), (11000, 20)]),
    ):
        for offset, length in gaps:
            if offset + length < n:
                frame.iloc[offset : offset + length, frame.columns.get_loc(column)] = np.nan

    missing_target = max(0.0, 1.0 - target_completeness)
    already = frame["PM 2.5 ug/m3"].isna().mean()
    extra = max(0.0, missing_target - already)
    if extra > 0:
        drop = rng.choice(n, size=int(extra * n), replace=False)
        frame.iloc[drop, frame.columns.get_loc("PM 2.5 ug/m3")] = np.nan

    for position in (500, 4321, 8888):
        if position < n:
            frame.iloc[position, frame.columns.get_loc("PM 2.5 ug/m3")] = 2000.0
    return frame


def write_station_csv(frame: pd.DataFrame, path, *, duplicate_rows: int = 0) -> None:
    out = frame.copy()
    out.index = out.index.strftime("%m/%d/%Y %H:%M")
    out.index.name = "Date"
    if duplicate_rows:
        out = pd.concat([out, out.iloc[:duplicate_rows]])
    out.to_csv(path)


@pytest.fixture(scope="session")
def raw_archive(tmp_path_factory):
    """A miniature three-station archive with realistic defects."""
    root = tmp_path_factory.mktemp("raw_archive")
    write_station_csv(synthetic_station(seed=1), root / "Alpha.csv")
    write_station_csv(synthetic_station(seed=2), root / "Beta.csv", duplicate_rows=24)
    write_station_csv(synthetic_station(seed=3), root / "Gamma.csv")
    return root


@pytest.fixture
def smoke_config(raw_archive, tmp_path):
    from pathlib import Path

    from pm25forecast.config import load_config

    repo_root = Path(__file__).resolve().parents[1]
    return load_config(
        repo_root / "configs" / "quick_test.yaml",
        overrides={
            "paths": {"raw_dir": str(raw_archive), "work_dir": str(tmp_path / "artifacts")}
        },
    )
