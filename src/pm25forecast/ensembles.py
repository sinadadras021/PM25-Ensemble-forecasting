"""Six Table 2 ensembles fitted exclusively on held-out base predictions."""

from pathlib import Path
import numpy as np
import joblib
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from .models import build_meta, callbacks
from .config import BASE_NAMES, write_json


def flat_design(members, context):
    """Flatten cycle/lead rows while retaining four individual member forecasts."""
    members = np.asarray(members, dtype="float64")
    context = np.asarray(context, dtype="float64")
    if (
        members.ndim != 3
        or members.shape[1:] != (48, 4)
        or context.shape[:2] != members.shape[:2]
    ):
        raise ValueError("Meta input shape mismatch.")
    design = np.concatenate([members, context], axis=2).reshape(
        -1, 4 + context.shape[2]
    )
    if not np.isfinite(design).all():
        raise ValueError("Nonfinite meta features; no silent zero filling is allowed.")
    return design


def fit_ensembles(members, y, context, hours, train_cycles, valid_cycles, cfg, folder):
    from xgboost import XGBRegressor

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    design = flat_design(members, context)
    truth = np.asarray(y).ravel()
    mask = np.isfinite(truth)
    train = np.repeat(train_cycles, 48) & mask
    valid = np.repeat(valid_cycles, 48) & mask
    if not train.any() or not valid.any():
        raise ValueError("Empty chronological 80/20 meta split.")
    scaler = MinMaxScaler().fit(design[train])
    scaled = scaler.transform(design).astype("float32")
    mlp = build_meta(design.shape[1], cfg)
    hist = mlp.fit(
        scaled[train],
        truth[train],
        validation_data=(scaled[valid], truth[valid]),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=False,
        callbacks=callbacks(cfg),
        verbose=0,
    )
    mlp.save(str(folder / "mlp_meta.h5"))
    joblib.dump(scaler, folder / "meta_scaler.joblib")
    write_json(folder / "MLP_training_history.json", hist.history)
    xgb = XGBRegressor(
        n_estimators=cfg.xgb_trees,
        max_depth=cfg.xgb_depth,
        learning_rate=cfg.xgb_learning_rate,
        objective="reg:squarederror",
        random_state=cfg.seed,
        n_jobs=-1,
    )
    xgb.fit(scaled[train], truth[train])
    matrix = members.reshape(-1, 4).astype("float64")
    X = matrix[mask]
    target = truth[mask]

    def objective(w):
        return np.mean((X @ w - target) ** 2)

    def gradient(w):
        return 2 * X.T @ (X @ w - target) / len(target)

    result = minimize(
        objective,
        np.full(4, 0.25),
        jac=gradient,
        method="SLSQP",
        bounds=[(0, 1)] * 4,
        constraints=[
            {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(4)}
        ],
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if (
        not result.success
        or not np.isfinite(result.x).all()
        or abs(result.x.sum() - 1) > 1e-6
    ):
        raise RuntimeError(
            f"PWE optimization failed; do not relabel a uniform fallback as PWE: {result.message}"
        )
    hm = {}
    flat_hours = hours.ravel()
    average = matrix.mean(axis=1)
    for hour in range(24):
        rows = mask & (flat_hours == hour)
        if rows.sum() < 2:
            raise ValueError(f"H-MOS hour {hour} has fewer than two observations.")
        hm[hour] = LinearRegression().fit(average[rows, None], truth[rows])
    labels = np.argmin(np.abs(matrix[train] - truth[train, None]), axis=1)
    # Table 2 specifies Base Preds + Met + Cyc + Offset for SMS. P091 omits
    # some of these; that manuscript inconsistency is documented in the report.
    sms = RandomForestClassifier(
        n_estimators=cfg.sms_trees,
        max_depth=cfg.sms_depth,
        criterion="gini",
        random_state=cfg.seed,
        n_jobs=-1,
    ).fit(design[train], labels)
    state = dict(
        pwe=result.x,
        xgb=xgb,
        h_mos=hm,
        sms=sms,
        feature_names=BASE_NAMES + cfg.context,
        meta_train_rows=int(train.sum()),
        meta_validation_rows=int(valid.sum()),
        fit_source="base-held-out-validation-only",
        MLP_seed=cfg.seed,
        pwe_and_hmos_fit_rows=int(mask.sum()),
        meta_train_ratio=cfg.meta_train_ratio,
    )
    joblib.dump(state, folder / "ensembles.joblib")
    write_json(folder / "PWE_weights.json", dict(zip(BASE_NAMES, map(float, result.x))))
    write_json(
        folder / "meta_protocol.json",
        {k: v for k, v in state.items() if k not in {"xgb", "h_mos", "sms", "pwe"}},
    )
    return state


def predict_ensembles(members, context, hours, folder, *, mlp_model=None):
    import tensorflow as tf

    folder = Path(folder)
    state = joblib.load(folder / "ensembles.joblib")
    scaler = joblib.load(folder / "meta_scaler.joblib")
    if mlp_model is None:
        mlp_model = tf.keras.models.load_model(
            str(folder / "mlp_meta.h5"), compile=False
        )
    design = flat_design(members, context)
    scaled = scaler.transform(design).astype("float32")
    shape = members.shape[:2]
    matrix = members.reshape(-1, 4)
    average = matrix.mean(axis=1)
    corrected = np.empty(len(matrix))
    local_hours = hours.ravel()
    for h, model in state["h_mos"].items():
        rows = local_hours == h
        corrected[rows] = model.predict(average[rows, None])
    selection = state["sms"].predict(design).astype(int)
    out = dict(
        sa=average,
        pwe=matrix @ state["pwe"],
        mlp_meta=mlp_model.predict(scaled, batch_size=32, verbose=0).ravel(),
        xgb_meta=state["xgb"].predict(scaled),
        h_mos=corrected,
        sms=matrix[np.arange(len(matrix)), selection],
    )
    for name, value in out.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{name} returned nonfinite predictions.")
        out[name] = np.maximum(value, 0).reshape(shape)
    return out
