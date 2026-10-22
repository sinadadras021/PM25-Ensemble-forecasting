"""Tables from one prediction population; no stored manuscript numbers are embedded."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from .config import BASE_NAMES, ENSEMBLE_NAMES, write_json, sha256

ALL_MODELS = BASE_NAMES + ENSEMBLE_NAMES


def metrics(y, p):
    """Compute continuous scores on observed finite prediction/target pairs."""
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    if y.shape != p.shape:
        raise ValueError("Metric array shapes differ.")
    good = np.isfinite(y) & np.isfinite(p)
    y = y[good]
    p = p[good]
    if not len(y):
        return dict(n=0, RMSE=np.nan, MAE=np.nan, R=np.nan, IOA=np.nan, NAED=np.nan)
    error = p - y
    denom = np.sum((np.abs(p - y.mean()) + np.abs(y - y.mean())) ** 2)
    return dict(
        n=len(y),
        RMSE=float(np.sqrt(np.mean(error**2))),
        MAE=float(np.mean(abs(error))),
        R=(
            float(np.corrcoef(y, p)[0, 1])
            if len(y) > 1 and y.std() > 0 and p.std() > 0
            else np.nan
        ),
        IOA=float(1 - np.sum(error**2) / denom) if denom > 0 else np.nan,
        NAED=(
            float(np.std(abs(error), ddof=0) / y.mean() * 100)
            if y.mean() > 0
            else np.nan
        ),
    )


def warning(y, p, threshold=35):
    """Count threshold events and return standard contingency-table scores."""
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    good = np.isfinite(y) & np.isfinite(p)
    y = y[good] > threshold
    p = p[good] > threshold
    tp = int((y & p).sum())
    fp = int((~y & p).sum())
    fn = int((y & ~p).sum())
    tn = int((~y & ~p).sum())

    def ratio(a, b):
        return a / b if b else np.nan

    return dict(
        n=len(y),
        TP=tp,
        FP=fp,
        FN=fn,
        TN=tn,
        POD=ratio(tp, tp + fn),
        Precision=ratio(tp, tp + fp),
        FAR=ratio(fp, tp + fp),
        FPR=ratio(fp, fp + tn),
        Specificity=ratio(tn, tn + fp),
        F1=ratio(2 * tp, 2 * tp + fp + fn),
        CSI=ratio(tp, tp + fp + fn),
        HSS=ratio(
            2 * (tp * tn - fp * fn), (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        ),
        Frequency_Bias=ratio(tp + fp, tp + fn),
        Accuracy=ratio(tp + tn, len(y)),
    )


def daily_predictions(frame, cfg):
    frame = frame.copy()
    frame["day"] = (frame.lead - 1) // 24 + 1
    if frame.duplicated(["station", "forecast_start", "lead"]).any():
        raise ValueError("Duplicate station/cycle/lead predictions.")
    if not frame.lead.between(1, 48).all():
        raise ValueError("Lead must be 1..48, not a mixture of lead conventions.")
    computed = (frame.DateTime - frame.forecast_start).dt.total_seconds() / 3600 + 1
    if not np.array_equal(computed, frame.lead):
        raise ValueError("Prediction timestamps and lead indices disagree.")
    out = []
    for (station, start, day), g in frame.groupby(
        ["station", "forecast_start", "day"], sort=True
    ):
        if len(g) != 24 or g.actual.notna().sum() < cfg.min_target_hours_per_day:
            continue
        row = dict(
            station=station,
            forecast_start=start,
            day=day,
            Date=g.DateTime.min().tz_convert(cfg.pollutant_timezone).date(),
            n_observed=int(g.actual.notna().sum()),
            actual=float(g.actual.mean()),
        )
        for model in ALL_MODELS:
            if not np.isfinite(g[model]).all():
                raise ValueError(f"{model}: incomplete forecast day.")
            row[model] = float(g[model].mean())
        out.append(row)
    return pd.DataFrame(out)


def dm_test(d):
    """Two-sided DM test with Bartlett HAC variance and the h=1 HLN correction."""
    d = np.asarray(d, dtype="float64")
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return dict(n_cycles=n, DM=np.nan, p_value=np.nan, HAC_bandwidth_cycles=np.nan)
    z = d - d.mean()
    lag = min(n - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    variance = np.dot(z, z) / n + 2 * sum(
        (1 - k / (lag + 1)) * np.dot(z[k:], z[:-k]) / n for k in range(1, lag + 1)
    )
    statistic = (
        d.mean() / np.sqrt(variance / n) * np.sqrt((n - 1) / n)
        if variance > 0
        else np.nan
    )
    return dict(
        n_cycles=n,
        DM=statistic,
        p_value=2 * student_t.sf(abs(statistic), n - 1),
        HAC_bandwidth_cycles=lag,
        mean_loss_difference=float(d.mean()),
        HLN_effective_horizon_cycles=1,
    )


def holm(table):
    table = table.copy()
    p = table.p_value.to_numpy()
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    table["p_Holm"] = np.nan
    if len(order):
        table.loc[table.index[order], "p_Holm"] = np.minimum(
            1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(order))))
        )
    table["significant_005"] = table.p_Holm.lt(0.05).where(table.p_Holm.notna())
    return table


def analyze(cfg):
    """Derive all daily tables from one hash-verified final prediction file."""
    import json
    from .pipeline import require_training, _identity

    require_training(cfg)
    path = cfg.out / "test_predictions.csv"
    manifest = json.loads(
        (cfg.out / "predictions_complete.json").read_text(encoding="utf-8")
    )
    if manifest["identity"] != _identity(cfg) or sha256(path) != manifest["sha256"]:
        raise ValueError(
            "Prediction file does not match the current trained model artifacts."
        )
    frame = pd.read_csv(path, parse_dates=["forecast_start", "DateTime"])
    daily = daily_predictions(frame, cfg)
    if daily.empty:
        raise ValueError("No verified daily records.")
    daily.to_csv(cfg.out / "daily_predictions.csv", index=False)
    rows = []
    for (station, day), g in daily.groupby(["station", "day"]):
        for model in ALL_MODELS:
            rows.append(
                dict(
                    station=station, day=day, model=model, **metrics(g.actual, g[model])
                )
            )
    station = pd.DataFrame(rows)
    station.to_csv(cfg.out / "station_metrics.csv", index=False)
    table4 = (
        station.groupby(["day", "model"])
        .agg(
            stations=("station", "nunique"),
            RMSE=("RMSE", "mean"),
            MAE=("MAE", "mean"),
            R=("R", "mean"),
            IOA=("IOA", "mean"),
            NAED=("NAED", "mean"),
        )
        .reset_index()
    )
    table4.to_csv(cfg.out / "Table4.csv", index=False)
    # Only complete station panels enter a city average or network loss sequence.
    groups = daily.groupby(["forecast_start", "day"])
    complete = groups.filter(lambda g: g.station.nunique() == cfg.expected_stations)
    city = (
        complete.groupby(["forecast_start", "day"])[["actual"] + ALL_MODELS]
        .mean()
        .reset_index()
    )
    city.to_csv(cfg.out / "city_daily_predictions.csv", index=False)
    warnings = []
    for day, g in city.groupby("day"):
        for model in ALL_MODELS:
            warnings.append(
                dict(day=day, model=model, **warning(g.actual, g[model], cfg.threshold))
            )
    pd.DataFrame(warnings).to_csv(cfg.out / "city_warning_metrics.csv", index=False)
    dm = []
    for day in [1, 2]:
        g = complete[complete.day == day]
        for model in ["bi_gru", "pwe", "xgb_meta", "sms"]:
            differences = (g.mlp_meta - g.actual) ** 2 - (g[model] - g.actual) ** 2
            loss = (
                pd.DataFrame(dict(start=g.forecast_start, value=differences))
                .groupby("start")
                .value.mean()
            )
            dm.append(
                dict(
                    day=day,
                    comparison=f"mlp_meta vs {model}",
                    **dm_test(loss.to_numpy()),
                )
            )
    holm(pd.DataFrame(dm)).to_csv(cfg.out / "DM_Holm.csv", index=False)
    episodes = daily[
        daily.Date.astype(str).isin(
            ["2024-09-01", "2024-09-02", "2024-12-12", "2024-12-13"]
        )
    ].copy()
    episodes["MLP_error"] = episodes.mlp_meta - episodes.actual
    episodes.to_csv(cfg.out / "episode_station_validation.csv", index=False)
    write_json(
        cfg.out / "analysis_population.json",
        dict(
            prediction_sha256=sha256(path),
            Table4="arithmetic mean of station daily metrics; not RMSE of city means",
            warnings="city mean over complete station panels only; event is strictly > threshold",
            DM="station-averaged daily squared-loss difference by nonoverlapping issue cycle; HAC+HLN h1; Holm family8",
            daily_records=len(daily),
            city_records=len(city),
            test_cycles_not_hardcoded=True,
        ),
    )
    return table4
