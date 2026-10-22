"""Every computation requires an explicit command; importing does not run research."""

import argparse
import json
from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Current-manuscript pooled PM2.5 pipeline; historical results cannot be reused."
    )
    parser.add_argument(
        "command",
        choices=[
            "validate-config",
            "prepare",
            "train",
            "predict",
            "analyze",
            "explain",
            "plot-shap",
            "ablation",
            "grid-search",
        ],
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "validate-config":
        cfg.validate(require_data=True)
        print(
            json.dumps(
                dict(
                    valid=True,
                    base_inputs=cfg.features,
                    meta_inputs=["bi_gru", "bi_lstm", "gru", "lstm"] + cfg.context,
                ),
                ensure_ascii=False,
            )
        )
        return
    from . import pipeline

    action = {
        "prepare": pipeline.prepare,
        "train": pipeline.train,
        "predict": pipeline.predict,
        "grid-search": pipeline.grid_search,
        "ablation": pipeline.ablations,
    }
    if args.command == "analyze":
        from .analysis import analyze

        analyze(cfg)
    elif args.command == "explain":
        from .explain import explain

        explain(cfg)
    elif args.command == "plot-shap":
        from .plots import plot_shap

        plot_shap(cfg)
    else:
        action[args.command](cfg)


if __name__ == "__main__":
    main()
