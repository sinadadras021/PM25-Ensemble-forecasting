"""Three separate scientific SHAP plots, only after this version is actually run."""

import numpy as np
import pandas as pd


def plot_shap(cfg):
    import json
    from .pipeline import _identity
    import shap
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    folder = cfg.out / "SHAP"
    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    if (
        protocol.get("identity") != _identity(cfg)
        or protocol.get("model_scope") != "pooled"
    ):
        raise ValueError(
            "These SHAP files are not explanations of the current pooled manuscript model."
        )
    data = np.load(folder / "MLP.npz", allow_pickle=False)
    values = data["shap"]
    x = data["x"]
    names = data["feature_names"].astype(str)
    if values.shape != x.shape or values.shape[1] != len(names):
        raise ValueError("SHAP/X/feature shape mismatch.")
    plt.rcParams["font.family"] = "Arial"
    label_font = FontProperties(family="Arial", size=30, weight="bold")
    tick_font = FontProperties(family="Arial", size=27, weight="bold")

    def style(ax):
        ax.xaxis.label.set_fontproperties(label_font)
        ax.yaxis.label.set_fontproperties(label_font)
        for label in (
            [ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            if label not in [ax.xaxis.label, ax.yaxis.label]:
                label.set_fontproperties(tick_font)
            label.set_color("black")

    def save(fig, name):
        for ext in ["png", "jpg", "pdf"]:
            opts = {"pil_kwargs": {"quality": 95}} if ext == "jpg" else {}
            fig.savefig(figures / f"{name}.{ext}", dpi=600, bbox_inches=None, **opts)
        plt.close(fig)

    importance = abs(values).mean(axis=0)
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(14, 9.8))
    bars = ax.barh(names[order], importance[order], color="#1f77b4", edgecolor="black")
    ax.set_xlabel("Mean absolute SHAP value (µg/m³)")
    ax.set_xlim(0, float(importance.max()) * 1.18)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    style(ax)
    for bar, value in zip(bars, importance[order]):
        ax.text(
            value + importance.max() * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=23,
            fontweight="bold",
        )
    fig.tight_layout()
    position = ax.get_position().bounds
    save(fig, "MLP_Meta_SHAP_Bar")
    np.random.seed(cfg.seed)
    fig = plt.figure(figsize=(14, 9.8))
    shap.summary_plot(
        values,
        features=x,
        feature_names=names,
        max_display=len(names),
        show=False,
        plot_size=(14, 9.8),
    )
    ax = plt.gca()
    ax.set_xlabel("SHAP value (µg/m³)")
    style(ax)
    for collection in ax.collections:
        if hasattr(collection, "set_sizes"):
            collection.set_sizes([150])
    left, bottom, _, height = position
    ax.set_position([left, bottom, 0.55, height])
    cbar = fig.axes[-1]
    cbar.set_position([0.90, bottom, 0.010, height])
    cbar.set_ylabel("Feature value")
    style(cbar)
    cbar.yaxis.set_label_coords(3, 0.5)
    save(fig, "MLP_Meta_SHAP_Beeswarm")
    base = pd.read_csv(folder / "BiGRU_importance.csv").pivot(
        index="feature", columns="day", values="mean_sum_abs_lag_SHAP"
    )
    base = base.loc[base.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(14, 9.8))
    y = np.arange(len(base))
    h = 0.36
    ax.barh(
        y - h / 2, base[1], height=h, color="#1f77b4", edgecolor="black", label="Day 1"
    )
    ax.barh(
        y + h / 2, base[2], height=h, color="#ff7f0e", edgecolor="black", label="Day 2"
    )
    ax.set_yticks(y)
    ax.set_yticklabels(base.index)
    ax.set_xlabel("Mean absolute SHAP value (µg/m³)")
    style(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, prop=tick_font, frameon=False
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "BiGRU_SHAP_Day1_Day2_Bar")
