"""SHAP of only Bi-GRU and MLP, using this version's own final artifacts."""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from .config import BASE_NAMES, write_json, sha256
from .pipeline import load_prepared, _identity, _predict_bases, require_training
from .windows import WindowPool
from .ensembles import flat_design


def explain(
    cfg,
    *,
    base_background_size=32,
    base_sample_size=24,
    gradient_samples=512,
    meta_background_size=48,
    meta_sample_size=32,
    kernel_samples=256,
):
    """Explain deployed models; smaller budgets support synthetic smoke tests.

    Defaults are the research budgets. A small test budget only establishes
    that the explainer executes, not that its attributions have converged.
    """
    if (
        min(
            base_background_size,
            base_sample_size,
            gradient_samples,
            meta_background_size,
            meta_sample_size,
            kernel_samples,
        )
        < 1
    ):
        raise ValueError("SHAP sampling budgets must be positive.")
    import tensorflow as tf
    import shap

    obj = load_prepared(cfg)
    require_training(cfg)
    train = WindowPool(obj, cfg, "train")
    test = WindowPool(obj, cfg, "test")
    rng = np.random.default_rng(cfg.seed)
    bg = rng.choice(len(train), min(base_background_size, len(train)), replace=False)
    samples = np.linspace(0, len(test) - 1, min(base_sample_size, len(test)), dtype=int)
    background = np.stack([train.sample(int(i))[0] for i in bg])
    inputs = np.stack([test.sample(int(i))[0] for i in samples])
    physical = (
        obj["xscaler"]
        .inverse_transform(inputs.reshape(-1, len(cfg.features)))
        .reshape(inputs.shape)
    )
    folder = cfg.out / "SHAP"
    folder.mkdir(exist_ok=True)
    model_path = cfg.out / "models" / "bi_gru.h5"
    base = tf.keras.models.load_model(str(model_path), compile=False)
    base_indices = test.exact_indices().iloc[samples].copy()
    base_indices["sample_index"] = samples
    base_indices.to_csv(folder / "BiGRU_sample_indices.csv", index=False)
    train.exact_indices().iloc[bg].to_csv(
        folder / "BiGRU_background_indices.csv", index=False
    )
    importances = []
    for day, start, end in [(1, 0, 24), (2, 24, 48)]:
        output = tf.keras.layers.Lambda(
            lambda x, a=start, b=end: tf.reduce_mean(
                tf.maximum(
                    x[:, a:b, 0] / float(obj["yscaler"].scale_[0])
                    - float(obj["yscaler"].min_[0] / obj["yscaler"].scale_[0]),
                    0,
                ),
                axis=1,
                keepdims=True,
            )
        )(base.output)
        predictor = tf.keras.Model(base.input, output)
        values = shap.GradientExplainer(predictor, background).shap_values(
            inputs, nsamples=gradient_samples, rseed=cfg.seed
        )
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values)
        if values.ndim == 4:
            values = values[:, :, :, 0]
        expected = float(predictor(background, training=False).numpy().mean())
        prediction = predictor(inputs, training=False).numpy().ravel()
        if values.shape != inputs.shape or not np.isfinite(values).all():
            raise ValueError("Invalid base SHAP tensor.")
        np.savez_compressed(
            folder / f"BiGRU_Day{day}.npz",
            lag_values=values,
            grouped_signed_SHAP=values.sum(axis=1),
            physical_inputs=physical,
            scaled_inputs=inputs,
            background=background,
            feature_names=np.asarray(cfg.features),
            expected_value=expected,
            prediction=prediction,
            residual=prediction - expected - values.sum(axis=(1, 2)),
            sample_indices=samples,
            background_indices=bg,
            model_sha256=sha256(model_path),
            seed=cfg.seed,
            nsamples=gradient_samples,
        )
        for name, value in zip(cfg.features, abs(values).sum(axis=1).mean(axis=0)):
            importances.append(dict(day=day, feature=name, mean_sum_abs_lag_SHAP=value))
    pd.DataFrame(importances).to_csv(folder / "BiGRU_importance.csv", index=False)
    # Explain the deployed clipped MLP, not an in-sample surrogate.
    validation = np.load(cfg.out / "validation_meta_data.npz", allow_pickle=False)
    dev = flat_design(validation["members"], validation["context"])
    fit_rows = np.repeat(validation["train_cycles"], 48)
    bg_rows = rng.choice(
        np.flatnonzero(fit_rows),
        min(meta_background_size, int(fit_rows.sum())),
        replace=False,
    )
    # Sample pooled windows without constructing all raw sequence inputs at once.
    members = _predict_bases(test, cfg, cfg.out / "models")
    design = flat_design(members, test.context())
    rows = np.linspace(
        0, len(design) - 1, min(meta_sample_size, len(design)), dtype=int
    )
    scaler = joblib.load(cfg.out / "ensembles" / "meta_scaler.joblib")
    meta_path = cfg.out / "ensembles" / "mlp_meta.h5"
    meta = tf.keras.models.load_model(str(meta_path), compile=False)

    def predict(a):
        a = scaler.transform(a).astype("float32")
        return np.maximum(
            np.concatenate(
                [
                    meta(a[i : i + 32], training=False).numpy().ravel()
                    for i in range(0, len(a), 32)
                ]
            ),
            0,
        )

    np.random.seed(cfg.seed)
    explainer = shap.KernelExplainer(predict, dev[bg_rows])
    values = np.asarray(
        explainer.shap_values(design[rows], nsamples=kernel_samples, l1_reg=0)
    )
    if values.ndim == 3:
        values = values[:, :, 0]
    expected = float(np.asarray(explainer.expected_value).ravel()[0])
    prediction = predict(design[rows])
    residual = prediction - expected - values.sum(axis=1)
    if not np.isfinite(values).all() or np.max(abs(residual)) > 0.001:
        raise ValueError("MLP SHAP additivity check failed.")
    np.savez_compressed(
        folder / "MLP.npz",
        shap=values,
        x=design[rows],
        feature_names=np.asarray(BASE_NAMES + cfg.context),
        background=dev[bg_rows],
        sample_indices=rows,
        background_indices=bg_rows,
        expected_value=expected,
        prediction=prediction,
        residual=residual,
        model_sha256=sha256(meta_path),
        seed=cfg.seed,
        nsamples=kernel_samples,
    )
    selected = test.exact_indices().iloc[rows // 48].reset_index(drop=True)
    selected["lead"] = rows % 48 + 1
    selected["flat_row_index"] = rows
    selected.to_csv(folder / "MLP_sample_indices.csv", index=False)
    val_pool = WindowPool(obj, cfg, "val")
    selected = val_pool.exact_indices().iloc[bg_rows // 48].reset_index(drop=True)
    selected["lead"] = bg_rows % 48 + 1
    selected["flat_row_index"] = bg_rows
    selected.to_csv(folder / "MLP_background_indices.csv", index=False)
    write_json(
        folder / "protocol.json",
        dict(
            explained_models=["bi_gru", "mlp_meta"],
            MLP_inputs=BASE_NAMES + cfg.context,
            no_averaging_of_four_base_inputs=True,
            approximate_SHAP=True,
            not_historical_SHAP=True,
            identity=_identity(cfg),
            model_scope="pooled",
            BiGRU_model_sha256=sha256(model_path),
            MLP_model_sha256=sha256(meta_path),
        ),
    )
