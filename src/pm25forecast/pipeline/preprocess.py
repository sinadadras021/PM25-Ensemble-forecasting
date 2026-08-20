"""Stage 1 -- raw CSVs to model-ready station tables (paper §2.2).

Order of operations, each position load-bearing:

1. **Read and de-duplicate** -- before anything measures time.
2. **Reindex to a complete hourly grid** -- before anything counts hours.
3. **Split chronologically** -- 70 % base training, a 48-hour gap, 30 %
   validation, and 2024 held back entirely.
4. **Fit the IQR screen on the training block, apply it everywhere** -- and never
   to validation or test *labels*.
5. **Fit the KNN imputer on the training block, apply it everywhere.**
6. **Feature construction** -- wind components and cyclical hour encodings.
7. **Segmentation** -- long gaps become series breaks so no window can bridge them.
8. **Fit the min-max scaler on the training block**, then verify the claim rather
   than assert it.

Steps 3-5 in that order are the whole leakage argument. Splitting *before*
fitting anything is what makes "fitted on training data only" a structural
property of the pipeline rather than a promise in a comment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..data.imputation import TrainFittedKNNImputer, interpolate_short_gaps
from ..data.loading import StationFile, discover_stations, read_station_csv
from ..data.quality import (
    IQRScreen,
    QualityReport,
    assess_quality,
    drop_unusable_columns,
    reindex_to_hourly,
)
from ..data.segments import SEGMENT_COLUMN, build_segments, segment_summary
from ..features import build_feature_frame, select_feature_columns, target_input_name
from ..logging_utils import get_logger, section
from ..scaling import SplitScaler
from ..validation import IntegrityError, check_monotonic_hourly_index

logger = get_logger(__name__)

__all__ = ["run_preprocessing", "load_station_table", "split_blocks", "CLEANED_SUFFIX", "META_SUFFIX"]

CLEANED_SUFFIX = "_cleaned.parquet"
META_SUFFIX = "_meta.json"


# --------------------------------------------------------------------------- #
def split_blocks(table: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    """Cut the frame into train / val / test (paper §2.2).

    The 48-hour gap is not decoration. With a 48-hour input window and a 48-hour
    horizon, a window anchored at the last training hour occupies 95 further
    hours; without the purge the last training windows and the first validation
    windows would overlap almost completely, and early stopping would be reading
    data the model had just been fitted to.
    """
    dev_start, dev_end = cfg.split.development
    test_start, test_end = cfg.split.test

    development = table.loc[dev_start:dev_end]
    if development.empty:
        raise IntegrityError(
            f"split.development = {cfg.split.development} selects zero rows for this station."
        )

    cut = int(len(development) * cfg.split.base_train_ratio)
    gap = cfg.split.gap_hours
    if cut - gap < 1 or cut >= len(development):
        raise IntegrityError(
            f"base_train_ratio={cfg.split.base_train_ratio} with gap_hours={gap} leaves an "
            f"empty side on {len(development)} development rows."
        )

    blocks = {
        "train": development.iloc[: cut - gap],
        "val": development.iloc[cut:],
        "test": table.loc[test_start:test_end],
    }
    logger.info(
        "Split: train %d rows | gap %d h | val %d rows | test %d rows",
        len(blocks["train"]), gap, len(blocks["val"]), len(blocks["test"]),
    )
    return blocks


# --------------------------------------------------------------------------- #
def _preprocess_station(station: StationFile, cfg: Config) -> Dict[str, object]:
    """Steps 1-8 for one station."""
    name = station.name
    logger.info("-" * 78)
    logger.info("Station: %s", name)

    raw = read_station_csv(
        station,
        datetime_column=cfg.data.datetime_column,
        datetime_formats=cfg.data.datetime_formats,
    )
    gridded = reindex_to_hourly(raw, cfg.data.global_start, cfg.data.global_end, context=name)
    pruned = drop_unusable_columns(
        gridded, drop=cfg.data.drop_columns, protect=[cfg.data.target], context=name
    )

    report = assess_quality(
        pruned, station=name, target=cfg.data.target,
        min_completeness=cfg.data.min_target_completeness,
        min_rows=cfg.windows.input_hours + cfg.windows.horizon_hours + 100,
    )
    if not report.passed:
        return {"report": report}

    blocks = split_blocks(pruned, cfg)
    train_span = (blocks["train"].index.min(), blocks["train"].index.max())

    # -- 4. IQR screen, fitted on training data -------------------------------- #
    screen: Optional[IQRScreen] = None
    if cfg.outliers.enabled:
        screen_columns = cfg.outliers.columns or None
        screen = IQRScreen(k=cfg.outliers.k, columns=screen_columns).fit(blocks["train"])

        for split_name, block in blocks.items():
            if block.empty:
                continue
            # The target is screened in training only. Altering validation or test
            # labels would mean scoring the model against a truncated ground
            # truth -- and truncation removes exactly the episodes the model
            # exists to predict.
            protect = [] if (split_name == "train" and cfg.outliers.include_target) else [cfg.data.target]
            blocks[split_name], voided = screen.transform(
                block, protect=protect, context=f"{name}/{split_name}"
            )
            if split_name == "train":
                report.n_outliers = voided

    # -- 5. KNN imputation, fitted on training data ---------------------------- #
    if cfg.imputation.short_gap_hours:
        for split_name, block in blocks.items():
            if not block.empty:
                blocks[split_name] = interpolate_short_gaps(
                    block, limit=cfg.imputation.short_gap_hours, context=f"{name}/{split_name}"
                )

    imputer = TrainFittedKNNImputer(
        n_neighbors=cfg.imputation.n_neighbors, weights=cfg.imputation.weights
    ).fit(blocks["train"])
    imputer.assert_fitted_on(train_span)

    # The label and the encoder's view of the target are separated here.
    #
    #   <target>_input : gap-filled, always. A window must be complete to exist,
    #                    so one missing input hour would otherwise cost the whole
    #                    96-hour window around it.
    #   <target>       : the label. Reconstructed or observations-only depending
    #                    on imputation.impute_target.
    #
    # Keeping them apart is what lets the strict setting be usable at all: with a
    # single column, refusing to reconstruct the label also destroys the inputs.
    history_column = target_input_name(cfg.data.target)
    observed_labels = {
        split_name: block[cfg.data.target].copy()
        for split_name, block in blocks.items()
        if not block.empty
    }
    for split_name, block in blocks.items():
        if not block.empty:
            filled = imputer.transform(block, context=f"{name}/{split_name}")
            filled[history_column] = filled[cfg.data.target]
            if not cfg.imputation.impute_target:
                filled[cfg.data.target] = observed_labels[split_name]
            blocks[split_name] = filled

    if cfg.imputation.impute_target:
        reconstructed = sum(
            int(observed_labels[k].isna().sum()) for k in observed_labels
        )
        report.n_target_hours_imputed = reconstructed
        if reconstructed:
            logger.info(
                "%s: %d label hours were reconstructed by the imputer (%.2f%% of all rows). "
                "Set imputation.impute_target=false to score against observations only.",
                name, reconstructed,
                reconstructed / max(sum(len(b) for b in blocks.values()), 1) * 100,
            )

    # -- 6-7. features and segmentation --------------------------------------- #
    combined = pd.concat([blocks["train"], blocks["val"], blocks["test"]])
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    combined = build_feature_frame(combined, cfg, context=name)
    check_monotonic_hourly_index(combined, context=f"{name} after features", strict=False)

    # Feature selection excludes the raw label and uses the gap-filled input copy,
    # so the encoder can never read a value the metrics would then score against.
    feature_columns = select_feature_columns(
        combined, target=cfg.data.target, exclude=[SEGMENT_COLUMN]
    )
    segmented = build_segments(
        combined, feature_columns,
        max_gap_hours=cfg.windows.max_gap_hours, context=name,
    )
    segments = segment_summary(
        segmented, min_length=cfg.windows.input_hours + cfg.windows.horizon_hours
    )
    if segments.empty:
        report.passed = False
        report.reason = (
            f"no contiguous run reaches {cfg.windows.input_hours + cfg.windows.horizon_hours} "
            f"hours"
        )
        logger.warning("%s: %s", name, report.reason)
        return {"report": report}

    # -- 8. scaling ------------------------------------------------------------ #
    train_block = segmented.loc[train_span[0] : train_span[1]]
    scaler = SplitScaler(
        feature_columns, cfg.data.target,
        method=cfg.scaling.method, clip=cfg.scaling.clip_scaled,
    ).fit(train_block)
    scaler.assert_train_only(train_span)

    table_path = cfg.cleaned_dir / f"{station.slug}{CLEANED_SUFFIX}"
    segmented.to_parquet(table_path)
    scaler_path = scaler.save(cfg.cleaned_dir, prefix=f"scaler_{station.slug}")

    meta = {
        "station_name": name,
        "slug": station.slug,
        "source_file": str(station.path),
        "table": table_path.name,
        "scaler": scaler_path.name,
        "target": cfg.data.target,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "target_input_column": history_column,
        "target_hours_imputed": report.n_target_hours_imputed,
        "n_segments": int(len(segments)),
        "train_span": [str(train_span[0]), str(train_span[1])],
        "outliers_removed_train": report.n_outliers,
        "iqr_bounds": screen.table().to_dict("records") if screen is not None else [],
        "splits": {
            "development": list(cfg.split.development),
            "test": list(cfg.split.test),
            "base_train_ratio": cfg.split.base_train_ratio,
            "gap_hours": cfg.split.gap_hours,
        },
    }
    with (cfg.cleaned_dir / f"{station.slug}{META_SUFFIX}").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    logger.info("%s: saved %s (%d features)", name, table_path.name, len(feature_columns))
    meta["report"] = report
    return meta


# --------------------------------------------------------------------------- #
def run_preprocessing(cfg: Config) -> pd.DataFrame:
    """Preprocess every station and write the station quality report."""
    section(logger, "STAGE 1 / PREPROCESSING  (paper 2.2)")
    cfg.ensure_directories()

    stations = discover_stations(cfg.paths.raw_dir, cfg.data.station_glob)
    rows: List[Dict[str, object]] = []
    accepted = 0

    for station in stations:
        try:
            meta = _preprocess_station(station, cfg)
        except (IntegrityError, ValueError, KeyError) as exc:
            logger.error("%s: preprocessing failed -- %s", station.name, exc)
            rows.append({"station": station.name, "slug": station.slug, "accepted": False,
                         "reason": f"{type(exc).__name__}: {exc}"})
            continue

        report: QualityReport = meta["report"]  # type: ignore[assignment]
        row = report.as_row()
        row["slug"] = station.slug
        if report.passed:
            accepted += 1
            row.update({"n_features": meta["n_features"], "n_segments": meta["n_segments"]})
        rows.append(row)

    quality = pd.DataFrame(rows)
    destination = cfg.results_dir / "station_quality_report.csv"
    quality.to_csv(destination, index=False, encoding="utf-8-sig")

    section(logger, f"PREPROCESSING COMPLETE -- {accepted}/{len(stations)} stations admitted")
    if accepted == 0:
        raise IntegrityError(
            f"No station passed screening. Inspect {destination}; the 'reason' column "
            f"explains each rejection."
        )
    logger.info("Quality report: %s", destination)
    return quality


def load_station_table(cfg: Config, slug: str) -> Tuple[pd.DataFrame, Dict[str, object], SplitScaler]:
    """Reload one preprocessed station: table, metadata and fitted scaler."""
    meta_path = cfg.cleaned_dir / f"{slug}{META_SUFFIX}"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"No preprocessed station {slug!r} at {meta_path}. Run the preprocess stage first."
        )
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    table = pd.read_parquet(cfg.cleaned_dir / meta["table"])
    scaler = SplitScaler.load(cfg.cleaned_dir / meta["scaler"])
    return table, meta, scaler
