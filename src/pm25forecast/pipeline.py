"""Explicit stages; no legacy artifacts can be silently resumed as this protocol."""

from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import json
import platform
import time
import itertools
import joblib
import numpy as np
import pandas as pd
from .config import BASE_NAMES, write_json, sha256
from .data import prepare_network
from .windows import WindowPool, keras_sequence, meta_partition
from .models import build_base, callbacks
from .ensembles import fit_ensembles, predict_ensembles


def source_signature():
    # Plot typography and post-hoc analysis do not invalidate fitted weights.
    core = [
        "config.py",
        "data.py",
        "windows.py",
        "models.py",
        "ensembles.py",
        "pipeline.py",
    ]
    return {name: sha256(Path(__file__).parent / name) for name in core}


def _identity(cfg):
    return dict(config=cfg.fingerprint(), code=source_signature())


def require_training(cfg):
    """Refuse incomplete, changed or corrupted fitted model artifacts."""
    ready = json.loads((cfg.out / "training_complete.json").read_text(encoding="utf-8"))
    if ready["identity"] != _identity(cfg):
        raise ValueError(
            "Training artifacts belong to a different scientific protocol/configuration."
        )
    for name in BASE_NAMES:
        marker = json.loads(
            (cfg.out / "models" / f"{name}_complete.json").read_text(encoding="utf-8")
        )
        if (
            marker["identity"] != _identity(cfg)
            or sha256(cfg.out / "models" / f"{name}.h5") != marker["sha256"]
        ):
            raise ValueError(f"Invalid trained {name} artifact.")
    marker = json.loads(
        (cfg.out / "ensembles" / "complete.json").read_text(encoding="utf-8")
    )
    if marker["identity"] != _identity(cfg):
        raise ValueError("Invalid ensemble protocol.")
    for name, digest in marker["artifacts"].items():
        if sha256(cfg.out / "ensembles" / name) != digest:
            raise ValueError(f"Corrupted ensemble artifact: {name}")
    return ready


def prepare(cfg):
    """Prepare or validate a cache without modifying source observations."""
    cfg.validate(require_data=True)
    if (
        cfg.out.exists()
        and any(cfg.out.iterdir())
        and not (cfg.out / "prepared_manifest.json").exists()
    ):
        raise ValueError(
            "Output directory is nonempty and has no matching protocol manifest; choose a fresh directory."
        )
    cfg.out.mkdir(parents=True, exist_ok=True)
    cache = cfg.out / "prepared.joblib"
    meta = cfg.out / "prepared_manifest.json"
    if cache.exists() or meta.exists():
        if not (cache.exists() and meta.exists()):
            raise ValueError(
                "Incomplete preparation checkpoint; use a new output directory."
            )
        return load_prepared(cfg)
    obj = prepare_network(cfg)
    write_json(cfg.out / "config_used.json", asdict(cfg))
    obj["qc_counts"].to_csv(cfg.out / "QC_counts.csv", index=False)
    joblib.dump(obj, cache)
    write_json(
        meta,
        dict(
            **_identity(cfg),
            sources=obj["sources"],
            features=obj["features"],
            dataset_sha256=sha256(cache),
            model_scope="one pooled network per architecture across all stations",
            target_policy="observed labels; no validation/test target screening or imputation",
            centered_QC_latency_hours=1,
            context_policy="historical trajectory aligned lead-for-lag",
        ),
    )
    return obj


def load_prepared(cfg):
    """Validate config, code, input and cache hashes before loading trusted data."""
    manifest = json.loads(
        (cfg.out / "prepared_manifest.json").read_text(encoding="utf-8")
    )
    expected = _identity(cfg)
    if any(manifest[k] != v for k, v in expected.items()):
        raise ValueError(
            "Prepared artifacts belong to different code/config; choose a fresh output directory."
        )
    for item in manifest["sources"]:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("Input data changed after preparation.")
    if sha256(cfg.out / "prepared.joblib") != manifest["dataset_sha256"]:
        raise ValueError("Prepared cache integrity failure.")
    return joblib.load(cfg.out / "prepared.joblib")


def _predict_bases(pool, cfg, folder):
    import tensorflow as tf

    predictions = []
    for name in BASE_NAMES:
        marker = json.loads(
            (Path(folder) / f"{name}_complete.json").read_text(encoding="utf-8")
        )
        if (
            marker["identity"] != _identity(cfg)
            or sha256(Path(folder) / f"{name}.h5") != marker["sha256"]
        ):
            raise ValueError(f"{name}: checkpoint identity/hash mismatch.")
        model = tf.keras.models.load_model(
            str(Path(folder) / f"{name}.h5"), compile=False
        )
        batches = [
            model.predict(x, batch_size=cfg.batch_size, verbose=0)
            for x, _ in pool.batches(cfg.batch_size)
        ]
        scaled = np.concatenate(batches).reshape(len(pool), 48)
        values = (
            pool.prepared["yscaler"]
            .inverse_transform(scaled.reshape(-1, 1))
            .reshape(-1, 48)
        )
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: nonfinite prediction.")
        predictions.append(np.maximum(values, 0))
        del model
        tf.keras.backend.clear_session()
    return np.stack(predictions, axis=-1)


def train(cfg):
    """Fit four pooled bases and six ensembles; never evaluate test-year labels."""
    import tensorflow as tf

    cfg.validate(require_data=True)
    for gpu in tf.config.list_physical_devices("GPU"):
        if not tf.config.experimental.get_memory_growth(gpu):
            tf.config.experimental.set_memory_growth(gpu, True)
    obj = prepare(cfg)
    train_pool = WindowPool(obj, cfg, "train")
    val = WindowPool(obj, cfg, "val")
    modeldir = cfg.out / "models"
    modeldir.mkdir(exist_ok=True)
    identity = _identity(cfg)
    for pool in [train_pool, val]:
        pool.exact_indices().to_csv(
            cfg.out / f"{pool.split}_exact_indices.csv", index=False
        )
    for name in BASE_NAMES:
        path = modeldir / f"{name}.h5"
        marker = modeldir / f"{name}_complete.json"
        if marker.exists():
            record = json.loads(marker.read_text(encoding="utf-8"))
            if (
                record["identity"] != identity
                or not path.exists()
                or sha256(path) != record["sha256"]
            ):
                raise ValueError(
                    f"{name}: incompatible/corrupt checkpoint; choose a new output directory."
                )
            continue
        if path.exists():
            raise ValueError(
                f"Uncommitted {path}; preserve it and use a new output directory."
            )
        model = build_base(name, len(cfg.features), cfg)
        started = time.perf_counter()
        history = model.fit(
            keras_sequence(train_pool, cfg.batch_size),
            validation_data=keras_sequence(val, cfg.batch_size),
            epochs=cfg.epochs,
            shuffle=False,
            callbacks=callbacks(cfg),
            verbose=0,
        )
        if (
            not np.isfinite(history.history["loss"]).all()
            or not np.isfinite(history.history["val_loss"]).all()
        ):
            raise ValueError("Nonfinite training; checkpoint not marked complete.")
        model.save(str(path))
        write_json(modeldir / f"{name}_history.json", history.history)
        write_json(
            marker,
            dict(
                identity=identity,
                sha256=sha256(path),
                seconds=time.perf_counter() - started,
                parameters=model.count_params(),
                loss="MSE on observed targets",
                pooled_stations=len(obj["stations"]),
            ),
        )
        del model
        tf.keras.backend.clear_session()
    # No 2024 prediction or 2024 metric participates in fitting or selection.
    members = _predict_bases(val, cfg, modeldir)
    context = val.context()
    y = val.physical_y()
    hours = val.local_target_hours()
    fit_mask, validation_mask = meta_partition(val, cfg.meta_train_ratio)
    np.savez_compressed(
        cfg.out / "validation_meta_data.npz",
        members=members,
        context=context,
        y=y,
        hours=hours,
        train_cycles=fit_mask,
        validation_cycles=validation_mask,
        anchors=val.records[:, 0],
    )
    metadir = cfg.out / "ensembles"
    marker = metadir / "complete.json"
    if marker.exists():
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record["identity"] != identity:
            raise ValueError("Ensemble checkpoint configuration mismatch.")
        for path, digest in record["artifacts"].items():
            if sha256(metadir / path) != digest:
                raise ValueError("Ensemble checkpoint corrupted.")
    else:
        if metadir.exists() and any(metadir.iterdir()):
            raise ValueError(
                "Partial ensemble artifacts found; preserve them and choose a new output directory."
            )
        fit_ensembles(
            members, y, context, hours, fit_mask, validation_mask, cfg, metadir
        )
        write_json(
            marker,
            dict(
                identity=identity,
                artifacts={p.name: sha256(p) for p in metadir.iterdir() if p.is_file()},
                fit_source="held-out-validation-base-predictions",
                meta_fit_end=str(val.first_targets[fit_mask].max()),
                meta_validation_start=str(val.first_targets[validation_mask].min()),
            ),
        )
    write_json(
        cfg.out / "training_complete.json",
        dict(
            identity=identity,
            scope="pooled",
            base_models=4,
            ensemble_models=6,
            test_predictions_generated=False,
            seed=cfg.seed,
            python=platform.python_version(),
            tensorflow=tf.__version__,
        ),
    )


def predict(cfg):
    """Generate all admissible test cycles, including cycles with missing truth."""
    cfg.validate(require_data=True)
    obj = load_prepared(cfg)
    require_training(cfg)
    pool = WindowPool(obj, cfg, "test")
    pool.exact_indices().to_csv(cfg.out / "test_exact_indices.csv", index=False)
    members = _predict_bases(pool, cfg, cfg.out / "models")
    context = pool.context()
    y = pool.physical_y()
    hours = pool.local_target_hours()
    ensembles = predict_ensembles(members, context, hours, cfg.out / "ensembles")
    ns = pool.records[:, 0, None] + np.arange(48, dtype="int64")[None, :] * 3600 * 10**9
    out = pd.DataFrame(
        dict(
            station=np.repeat([pool.names[int(r[1])] for r in pool.records], 48),
            forecast_start=np.repeat(pool.first_targets, 48),
            DateTime=pd.to_datetime(ns.ravel(), utc=True),
            lead=np.tile(np.arange(1, 49), len(pool)),
            actual=y.ravel(),
        )
    )
    for i, name in enumerate(BASE_NAMES):
        out[name] = members[:, :, i].ravel()
    for name, value in ensembles.items():
        out[name] = value.ravel()
    out.to_csv(cfg.out / "test_predictions.csv", index=False)
    write_json(
        cfg.out / "predictions_complete.json",
        dict(
            identity=_identity(cfg),
            sha256=sha256(cfg.out / "test_predictions.csv"),
            forecast_admission_uses_future_targets=False,
        ),
    )
    write_json(
        cfg.out / "prediction_coverage.json",
        dict(
            cycles=len(pool),
            stations=len(pool.names),
            cycles_per_station=out.groupby("station")
            .forecast_start.nunique()
            .to_dict(),
            daily_records_per_station=(
                2 * out.groupby("station").forecast_start.nunique()
            ).to_dict(),
            test_end_inclusive=cfg.test_end,
            min_target_hours_per_day=cfg.min_target_hours_per_day,
            counted_from_artifacts=True,
            not_assumed_181=True,
        ),
    )
    return out


def grid_search(cfg):
    """Prospective validation-only grid; no unknown historical grid is invented."""
    allowed = {"learning_rate", "dropout"}
    if not cfg.grid or set(cfg.grid) - allowed:
        raise ValueError("Supply an explicit grid of learning_rate/dropout candidates.")
    if any(not values for values in cfg.grid.values()):
        raise ValueError("Empty grid axis.")
    candidates = list(itertools.product(*cfg.grid.values()))
    if len(candidates) < 2:
        raise ValueError(
            "A single fixed configuration is not an optimized grid search."
        )
    rows = []
    for i, values in enumerate(candidates):
        changes = dict(zip(cfg.grid, values))
        candidate = replace(
            cfg,
            **changes,
            grid={},
            output_dir=str(cfg.out / "grid" / f"candidate_{i:03}"),
        )
        train(candidate)
        scores = []
        for name in BASE_NAMES:
            hist = json.loads(
                (candidate.out / "models" / f"{name}_history.json").read_text(
                    encoding="utf-8"
                )
            )
            score = min(hist["val_loss"])
            scores.append(score)
            rows.append(
                dict(
                    candidate=i,
                    model=name,
                    validation_MSE=score,
                    **changes,
                    test_used=False,
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(cfg.out / "prospective_grid_results.csv", index=False)
    winner = int(table.groupby("candidate").validation_MSE.mean().idxmin())
    chosen = dict(zip(cfg.grid, candidates[winner]))
    write_json(
        cfg.out / "grid_selected_config.json",
        asdict(
            replace(
                cfg,
                **chosen,
                grid={},
                output_dir=str(cfg.out / "grid" / f"candidate_{winner:03}"),
            )
        ),
    )
    write_json(
        cfg.out / "grid_provenance.json",
        dict(
            candidates=cfg.grid,
            selection_metric="validation scaled observed MSE",
            historical_grid_reproduced=False,
            provenance=cfg.grid_provenance,
        ),
    )


def ablations(cfg):
    """Full baseline references exactly the full pipeline artifact; no refitting of Full."""
    from .analysis import analyze

    if cfg.exclude_groups:
        raise ValueError("Ablations must start from the unablated full configuration.")
    if not (cfg.out / "test_predictions.csv").exists():
        raise ValueError("Run the final full model before ablations.")
    baseline = analyze(cfg)
    baseline = baseline[(baseline.model == "mlp_meta")].copy()
    baseline["configuration"] = "full"
    tables = [baseline]
    for name, group in [
        ("no_meteorology", "meteorology"),
        ("no_cyclical_time", "cyclical"),
        ("no_forecast_offset", "offset"),
    ]:
        variant = replace(
            cfg,
            exclude_groups=[group],
            output_dir=str(cfg.out / "ablations" / name),
            grid={},
        )
        train(variant)
        predict(variant)
        table = analyze(variant)
        table = table[table.model == "mlp_meta"].copy()
        table["configuration"] = name
        tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(
        cfg.out / "ablation_results.csv", index=False
    )
    write_json(
        cfg.out / "ablation_baseline_identity.json",
        dict(
            source="same Table4.csv full-model rows",
            full_prediction_sha256=sha256(cfg.out / "test_predictions.csv"),
            full_refitted=False,
        ),
    )
