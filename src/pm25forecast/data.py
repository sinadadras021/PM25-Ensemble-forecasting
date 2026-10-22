"""Training-calibrated four-condition QC and issue-time-safe history features."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from .config import POLLUTANTS, METEOROLOGY, CYCLICAL, sha256


def read_frame(path, cfg, timezone):
    frame = pd.read_csv(path)
    if cfg.datetime_column not in frame:
        raise ValueError(f"{path}: missing {cfg.datetime_column!r}.")
    index = pd.DatetimeIndex(
        pd.to_datetime(
            frame.pop(cfg.datetime_column), format=cfg.datetime_format, errors="raise"
        )
    )
    if index.tz is None:
        index = index.tz_localize(
            timezone,
            ambiguous=cfg.ambiguous_local_time,
            nonexistent=cfg.nonexistent_local_time,
        )
    index = index.tz_convert("UTC")
    if index.has_duplicates or index.hasnans:
        raise ValueError(
            f"{path}: duplicate/invalid timestamps; resolve their actual meaning first."
        )
    frame.index = index
    return (
        frame.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .sort_index()
    )


def convert_weather(frame, cfg):
    frame = frame.copy()
    speed = 1 if cfg.wind_unit == "m/s" else 0.5144444444444445
    if {"U", "V"} <= set(frame):
        frame[["U", "V"]] = frame[["U", "V"]] * speed
    elif {"wind_speed", "wind_direction"} <= set(frame):
        radians = np.deg2rad(frame["wind_direction"])
        ws = frame["wind_speed"] * speed
        frame["U"] = -ws * np.sin(radians)
        frame["V"] = -ws * np.cos(radians)
    else:
        raise ValueError(
            "Provide U,V or wind_speed,wind_direction with declared units."
        )
    if not set(METEOROLOGY) <= set(frame):
        raise ValueError(f"Missing weather columns: {set(METEOROLOGY)-set(frame)}")
    frame["vsby"] = (
        frame["vsby"] * {"m": 0.001, "km": 1, "mile": 1.609344}[cfg.visibility_unit]
    )
    frame["alti"] = frame["alti"] * (1 if cfg.pressure_unit == "hPa" else 33.8638866667)
    return frame[METEOROLOGY]


def load_network(cfg):
    """Load station observations on one UTC grid without joining future weather."""
    cfg.validate(require_data=True)
    met_path = Path(cfg.meteorology_file).resolve() if cfg.meteorology_file else None
    files = [
        p for p in sorted(Path(cfg.raw_dir).glob("*.csv")) if p.resolve() != met_path
    ]
    if len(files) != cfg.expected_stations:
        raise ValueError(
            f"Expected {cfg.expected_stations} stations, found {len(files)}."
        )
    start = cfg.timestamp(cfg.development_start)
    end = cfg.timestamp(cfg.test_end).floor("h")
    # Retain the station clock's minute phase (Tehran observations are UTC xx:30).
    grid = pd.date_range(start, end + pd.Timedelta(hours=1), freq="h")
    grid = grid[grid <= cfg.timestamp(cfg.test_end)]
    met = (
        convert_weather(read_frame(met_path, cfg, cfg.meteorology_timezone), cfg)
        if met_path
        else None
    )
    if met is not None:
        met.index = met.index + pd.Timedelta(
            minutes=cfg.meteorology_reporting_delay_minutes
        )
    frames = {}
    sources = []
    for path in files:
        raw = read_frame(path, cfg, cfg.pollutant_timezone)
        if not set(POLLUTANTS) <= set(raw):
            raise ValueError(f"{path}: pollutant columns missing.")
        pollutants = raw[POLLUTANTS].mask(raw[POLLUTANTS] < 0).reindex(grid)
        if met is None:
            weather = convert_weather(raw, cfg).reindex(grid)
        else:
            # Backward ASOF: no meteorological report later than the pollutant hour.
            weather = pd.merge_asof(
                pd.DataFrame(index=grid),
                met,
                left_index=True,
                right_index=True,
                direction="backward",
                tolerance=pd.Timedelta(minutes=cfg.met_max_age_minutes),
            )
        frames[path.stem] = pollutants.join(weather)
        sources.append(dict(path=str(path.resolve()), sha256=sha256(path)))
    if met_path:
        sources.append(dict(path=str(met_path), sha256=sha256(met_path)))
    return frames, sources


def four_condition_screen(frames, cfg):
    """Exact simultaneous inequalities in manuscript P035, ddof=0.

    A centered three-point decision for hour t becomes available at t+1.
    Window construction therefore restores the last input row from unscreened
    observations. Other historical decisions are available by their issue time.
    Test/validation verification labels are never replaced by screened values.
    """
    names = sorted(frames)
    index = frames[names[0]].index
    calibration = {}
    flags = {
        name: pd.DataFrame(False, index=index, columns=POLLUTANTS) for name in names
    }
    counts = []
    for pollutant in POLLUTANTS:
        z = pd.DataFrame(index=index)
        for name in names:
            series = frames[name][pollutant]
            train = series.loc[cfg.timestamp(cfg.development_start) : cfg.train_end]
            mean = float(train.mean())
            std = float(train.std(ddof=0))
            if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
                raise ValueError(
                    f"{name}/{pollutant}: no variable observed training series for Z calibration."
                )
            calibration[f"{name}/{pollutant}"] = dict(
                mean=mean, std=std, fit_end=str(cfg.train_end)
            )
            z[name] = (series - mean) / std
        delta = z.diff()
        city = delta.mean(axis=1).where(
            delta.notna().sum(axis=1) >= cfg.z_min_city_stations
        )
        for name in names:
            rolling = z[name].rolling(3, center=True, min_periods=3).mean()
            condition = (
                (z[name].abs() > cfg.z_threshold)
                & (delta[name] > cfg.z_jump)
                & (z[name].div(rolling.where(rolling != 0)) > cfg.z_rolling_ratio)
                & (delta[name].div(city.where(city != 0)) > cfg.z_city_ratio)
            )
            if cfg.train_end in condition.index:
                condition.loc[cfg.train_end] = False
            flags[name][pollutant] = condition.fillna(False)
            for split, mask in split_masks(index, cfg).items():
                available = int(frames[name].loc[mask, pollutant].notna().sum())
                n = int(condition.loc[mask].sum())
                counts.append(
                    dict(
                        station=name,
                        pollutant=pollutant,
                        split=split,
                        observed=available,
                        flagged=n,
                        percent=100 * n / available if available else np.nan,
                    )
                )
    return flags, calibration, pd.DataFrame(counts)


def split_masks(index, cfg):
    return dict(
        train=(index >= cfg.timestamp(cfg.development_start))
        & (index <= cfg.train_end),
        val=(index >= cfg.cut) & (index <= cfg.timestamp(cfg.development_end)),
        test=(index >= cfg.timestamp(cfg.test_start))
        & (index <= cfg.timestamp(cfg.test_end)),
    )


def engineer(frame, cfg):
    frame = frame.copy()
    hours = frame.index.tz_convert(cfg.pollutant_timezone).hour
    frame["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    return frame[cfg.features]


def prepare_network(cfg):
    """Fit preprocessing on training data and preserve observed verification labels."""
    raw, sources = load_network(cfg)
    flags, zcal, qc = four_condition_screen(raw, cfg)
    prepared = {}
    xscaler = MinMaxScaler()
    yscaler = MinMaxScaler()
    for station, frame in raw.items():
        clean = frame.copy()
        clean[POLLUTANTS] = clean[POLLUTANTS].mask(flags[station])
        features = engineer(clean, cfg)
        unscreened = engineer(frame, cfg)
        train = split_masks(frame.index, cfg)["train"]
        # Centered QC for the last fitting hour cannot consume a validation/gap hour.
        boundary = np.flatnonzero(train)[-1]
        features.iloc[boundary] = unscreened.iloc[boundary]
        empty = features.loc[train].columns[features.loc[train].isna().all()].tolist()
        if empty:
            raise ValueError(f"{station}: all-missing training inputs: {empty}")
        imputer = KNNImputer(
            n_neighbors=cfg.knn_neighbors, weights=cfg.knn_weights
        ).fit(features.loc[train])
        finalized = imputer.transform(features).astype("float32")
        last = imputer.transform(unscreened).astype("float32")
        if not np.isfinite(finalized).all() or not np.isfinite(last).all():
            raise ValueError("Nonfinite imputed inputs.")
        labels = frame["PM 2.5 ug/m3"].to_numpy(dtype="float64")
        # Only training targets undergo the declared QC. No future labels are imputed.
        train_labels = labels.copy()
        train_labels[train & flags[station]["PM 2.5 ug/m3"].to_numpy()] = np.nan
        labels[train] = train_labels[train]
        xscaler.partial_fit(finalized[train])
        valid = labels[train & np.isfinite(labels)]
        if len(valid) == 0:
            raise ValueError(f"{station}: no observed training targets.")
        yscaler.partial_fit(valid.reshape(-1, 1))
        prepared[station] = dict(
            index=frame.index,
            physical=finalized,
            last_physical=last,
            y=labels,
            imputer=imputer,
        )
    for item in prepared.values():
        item["x"] = xscaler.transform(item["physical"]).astype("float32")
        item["last_x"] = xscaler.transform(item["last_physical"]).astype("float32")
        item["scaled_y"] = (
            yscaler.transform(item["y"].reshape(-1, 1)).ravel().astype("float32")
        )
    return dict(
        stations=prepared,
        xscaler=xscaler,
        yscaler=yscaler,
        features=cfg.features,
        sources=sources,
        z_calibration=zcal,
        qc_counts=qc,
        signature=cfg.fingerprint(),
    )
