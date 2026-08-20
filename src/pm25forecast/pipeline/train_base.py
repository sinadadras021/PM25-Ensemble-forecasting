"""Stage 2 -- train the four base learners, per station (paper §4.1).

Each station gets its own four models. Tehran's monitoring sites differ in
traffic exposure, altitude and local emission mix; a pooled model would learn the
city mean and systematically miss the station-level episodes that trigger
warnings.

The base learners' predictions on the **validation** block become the training
data for every ensemble in stage 3. That is why the validation split is 30 % here
rather than the usual 10 %: it is not only an early-stopping monitor, it is the
meta-learners' entire dataset.

Artefacts per station::

    02_base_models/<slug>/
        lstm.keras, gru.keras, bi_lstm.keras, bi_gru.keras
        member_predictions.npz     <- meta-features for stage 3
        training_history.json
        base_model_meta.json
"""

from __future__ import annotations

import gc
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..features import hour_offset_grid, target_input_name
from ..logging_utils import get_logger, section
from ..metrics import continuous_metrics
from ..models.base import build_base_model, compile_base_model
from ..models.losses import DEFAULT_LOSS_BY_MODEL
from ..seeding import environment_fingerprint, set_global_seed
from ..validation import (
    IntegrityError,
    check_finite,
    check_no_split_overlap_after_windowing,
    check_window_shapes,
)
from ..windowing import WindowSet, make_windows
from .preprocess import load_station_table, split_blocks

logger = get_logger(__name__)

__all__ = ["build_windows", "train_station", "run_base_training"]


def build_windows(
    blocks: Dict[str, pd.DataFrame], scaler, cfg: Config, *, station: str = ""
) -> Dict[str, WindowSet]:
    """Scale each block with the fitted scaler and window it.

    The scaler is applied, never re-fitted. Validation and test windows are
    anchored so that each covers exactly two calendar days, which is what makes
    the Day 1 / Day 2 evaluation of paper §4.4 well defined; training windows are
    unanchored because more samples help there and no daily aggregation is
    performed on them.
    """
    out: Dict[str, WindowSet] = {}
    for name, block in blocks.items():
        if block.empty:
            logger.warning("%s: split %r is empty.", station, name)
            continue
        scaled = scaler.transform(block, with_target=True)
        scaled["Segment_ID"] = block["Segment_ID"].to_numpy()
        out[name] = make_windows(
            scaled,
            scaler.feature_columns,
            input_hours=cfg.windows.input_hours,
            horizon_hours=cfg.windows.horizon_hours,
            stride=cfg.windows.stride.get(name, 1),
            min_label_completeness=cfg.windows.min_label_completeness,
            anchor_hour=cfg.windows.daily_anchor_hour if name != "train" else None,
            context=f"{station}/{name}",
        )

    check_no_split_overlap_after_windowing(
        {k: v.anchors for k, v in out.items()},
        input_hours=cfg.windows.input_hours,
        horizon_hours=cfg.windows.horizon_hours,
        purge_hours=0,
        context=station,
    )
    return out


# --------------------------------------------------------------------------- #
def train_station(slug: str, cfg: Config, *, overwrite: bool = False) -> Optional[Dict[str, object]]:
    """Train the configured architectures for one station and cache their
    validation and test predictions in ug/m3."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    table, meta, scaler = load_station_table(cfg, slug)
    station = str(meta["station_name"])
    logger.info("=" * 78)
    logger.info("Base models | %s", station)

    blocks = split_blocks(table, cfg)
    windows = build_windows(blocks, scaler, cfg, station=station)
    for name in ("train", "val", "test"):
        required = cfg.windows.min_windows.get(name, 0)
        have = len(windows[name]) if name in windows else 0
        if have < required:
            logger.warning(
                "%s: split %r has %d windows, minimum %d -- station skipped.",
                station, name, have, required,
            )
            return None

    train, val, test = windows["train"], windows["val"], windows["test"]
    check_window_shapes(
        train.X, train.y,
        input_hours=cfg.windows.input_hours, horizon_hours=cfg.windows.horizon_hours,
        n_features=len(scaler.feature_columns), context=f"{station}/train",
    )
    check_finite(train.X, name=f"{station}/train.X")

    model_dir = cfg.artifacts_dir / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    names = list(cfg.base_models.names)
    history_log: Dict[str, Dict[str, List[float]]] = {}
    predictions: Dict[str, Dict[str, np.ndarray]] = {"val": {}, "test": {}}
    validation_metrics: Dict[str, Dict[str, float]] = {}

    clip = cfg.evaluation.prediction_clip
    for name in names:
        destination = model_dir / f"{name}.keras"

        if destination.exists() and not overwrite:
            logger.info("%s: %s already trained; reusing.", station, name)
            model = tf.keras.models.load_model(destination, compile=False)
        else:
            tf.keras.backend.clear_session()
            set_global_seed(cfg.seed, deterministic=cfg.deterministic)
            model = build_base_model(
                name,
                input_steps=cfg.windows.input_hours,
                n_features=train.n_features,
                horizon=cfg.windows.horizon_hours,
                rnn_units=cfg.base_models.rnn_units,
                bi_units=cfg.base_models.bi_units,
                dropout=cfg.base_models.dropout,
            )
            compile_base_model(
                model, name=name,
                learning_rate=cfg.base_models.learning_rate,
                iwl_alpha=cfg.base_models.iwl_alpha,
                eel_power=cfg.base_models.eel_power,
            )
            logger.info(
                "%s: training %s (%s params, loss=%s) on %d windows, validating on %d.",
                station, name, f"{model.count_params():,}",
                DEFAULT_LOSS_BY_MODEL.get(name, "mse"), len(train), len(val),
            )
            history = model.fit(
                train.X, train.y,
                validation_data=(val.X, val.y),
                epochs=cfg.base_models.epochs,
                batch_size=cfg.base_models.batch_size,
                verbose=0,
                callbacks=[
                    EarlyStopping(
                        monitor="val_loss",
                        patience=cfg.base_models.early_stopping_patience,
                        restore_best_weights=True,
                    ),
                    ReduceLROnPlateau(
                        monitor="val_loss", factor=cfg.base_models.reduce_lr_factor,
                        patience=cfg.base_models.reduce_lr_patience,
                        min_lr=cfg.base_models.min_learning_rate,
                    ),
                ],
            )
            history_log[name] = {k: [float(x) for x in v] for k, v in history.history.items()}
            best_epoch = int(np.argmin(history.history["val_loss"])) + 1
            logger.info(
                "%s: %s stopped after %d epochs; best val_loss %.6f at epoch %d.",
                station, name, len(history.history["loss"]),
                float(np.min(history.history["val_loss"])), best_epoch,
            )
            model.save(destination)

        for split_name, window_set in (("val", val), ("test", test)):
            raw = model.predict(window_set.X, verbose=0)
            values = scaler.inverse_target(
                raw.reshape(len(window_set), cfg.windows.horizon_hours)
            )
            if clip is not None:
                values = np.clip(values, clip[0], clip[1])
            predictions[split_name][name] = values.astype(np.float32)

        truth_val = scaler.inverse_target(val.y)
        validation_metrics[name] = continuous_metrics(truth_val, predictions["val"][name])
        logger.info(
            "%s: %s validation RMSE %.3f | MAE %.3f | R %.3f",
            station, name, validation_metrics[name]["RMSE"],
            validation_metrics[name]["MAE"], validation_metrics[name]["R"],
        )
        del model
        gc.collect()

    # -- context features for the meta-learners (paper 4.2) ------------------- #
    # Meteorology and cyclical time at each *forecast* hour, plus Hour_offset.
    # Meteorology and clock features only: the meta-learners already receive the
    # member forecasts, so re-supplying pollutant levels would let them rebuild a
    # base model rather than learn how to combine them.
    context_columns = [
        c for c in scaler.feature_columns
        if not c.startswith(("PM", "O3", "CO", "NO", "SO2"))
    ]
    val_context = _forecast_context(val, table, scaler, context_columns, cfg)
    test_context = _forecast_context(test, table, scaler, context_columns, cfg)

    np.savez_compressed(
        model_dir / "member_predictions.npz",
        val_members=np.stack([predictions["val"][n] for n in names], axis=-1),
        val_y=scaler.inverse_target(val.y).astype(np.float32),
        val_context=val_context.astype(np.float32),
        val_anchors=np.array([pd.Timestamp(t).isoformat() for t in val.anchors], dtype="U32"),
        val_target_times=np.array(
            [[pd.Timestamp(t).isoformat() for t in row] for row in val.target_times], dtype="U32"
        ),
        test_members=np.stack([predictions["test"][n] for n in names], axis=-1),
        test_y=scaler.inverse_target(test.y).astype(np.float32),
        test_context=test_context.astype(np.float32),
        test_anchors=np.array([pd.Timestamp(t).isoformat() for t in test.anchors], dtype="U32"),
        test_target_times=np.array(
            [[pd.Timestamp(t).isoformat() for t in row] for row in test.target_times], dtype="U32"
        ),
        test_history=scaler.inverse_target(
            test.X[:, :, scaler.feature_columns.index(target_input_name(cfg.data.target))]
        ).astype(np.float32),
    )

    if history_log:
        with (model_dir / "training_history.json").open("w", encoding="utf-8") as handle:
            json.dump(history_log, handle, indent=2)

    payload = {
        "station_name": station,
        "slug": slug,
        "model_names": names,
        "feature_columns": scaler.feature_columns,
        "context_columns": context_columns + (["Hour_offset"] if cfg.features.hour_offset_feature else []),
        "n_features": train.n_features,
        "input_hours": cfg.windows.input_hours,
        "horizon_hours": cfg.windows.horizon_hours,
        "validation_metrics": {k: {m: float(v) for m, v in d.items()} for k, d in validation_metrics.items()},
        "n_train_windows": len(train),
        "n_val_windows": len(val),
        "n_test_windows": len(test),
        "environment": environment_fingerprint(),
    }
    with (model_dir / "base_model_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def _forecast_context(
    window_set: WindowSet, table: pd.DataFrame, scaler, columns: List[str], cfg: Config
) -> np.ndarray:
    """Meteorology + cyclical time at each forecast hour, plus ``Hour_offset``.

    Looked up by timestamp from the preprocessed table rather than carried
    through the encoder, because the meta-learners need conditions at the
    *forecast* hours, which lie outside the input window entirely.
    """
    n, horizon = len(window_set), cfg.windows.horizon_hours
    if n == 0:
        width = len(columns) + (1 if cfg.features.hour_offset_feature else 0)
        return np.empty((0, horizon, width), dtype=np.float32)

    times = pd.DatetimeIndex(pd.to_datetime(window_set.target_times.ravel()))
    lookup = table.reindex(times)
    block = lookup[columns].to_numpy(dtype=np.float64).reshape(n, horizon, len(columns))
    block = np.nan_to_num(block, nan=0.0)

    if cfg.features.hour_offset_feature:
        offsets = hour_offset_grid(n, horizon).astype(np.float64)[..., None]
        block = np.concatenate([block, offsets], axis=-1)
    return block


# --------------------------------------------------------------------------- #
def run_base_training(cfg: Config, *, overwrite: bool = False) -> List[str]:
    """Train base learners for every preprocessed station."""
    section(logger, "STAGE 2 / BASE LEARNERS  (paper 4.1)")
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)

    slugs = sorted(p.name.replace("_meta.json", "") for p in cfg.cleaned_dir.glob("*_meta.json"))
    if not slugs:
        raise FileNotFoundError(
            f"No preprocessed stations in {cfg.cleaned_dir}. Run the preprocess stage first."
        )

    trained: List[str] = []
    for slug in slugs:
        try:
            if train_station(slug, cfg, overwrite=overwrite) is not None:
                trained.append(slug)
        except (IntegrityError, ValueError, KeyError) as exc:
            logger.error("%s: base training failed -- %s", slug, exc)

    section(logger, f"BASE TRAINING COMPLETE -- {len(trained)}/{len(slugs)} stations")
    if not trained:
        raise IntegrityError("No station produced base models; see the errors above.")
    return trained
