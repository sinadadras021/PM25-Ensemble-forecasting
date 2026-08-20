"""Command-line interface.

Design decision worth stating, since the alternative gets proposed often: the
pipeline takes its paths from **CLI flags, a YAML config and environment
variables** -- never from ``input()`` prompts. Interactive prompts cannot be run
under CI, cannot be scripted across twenty stations, cannot be recorded in a
provenance file, and turn an unattended overnight run into a process blocked on
a cursor. Flags and files are the reproducible choice, and they are what a
reviewer expects to see.

Usage::

    pm25forecast run-all       --config configs/default.yaml --raw-dir /data/o3
    pm25forecast preprocess      --config configs/default.yaml
    pm25forecast train-base      --overwrite
    pm25forecast train-ensembles --overwrite
    pm25forecast evaluate
    pm25forecast ablation
    pm25forecast validate-config --config configs/default.yaml

Every subcommand accepts ``--raw-dir`` and ``--work-dir`` so a config file can be
shared unchanged between machines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import ConfigError, Config, dump_config, load_config
from .logging_utils import configure_logging, get_logger, section
from .seeding import environment_fingerprint, set_global_seed

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config. Defaults to built-in values, which you almost "
             "certainly want to override.",
    )
    parser.add_argument(
        "--raw-dir", type=str, default=None,
        help="Directory containing one CSV per station. Overrides paths.raw_dir.",
    )
    parser.add_argument(
        "--work-dir", type=str, default=None,
        help="Where every artefact is written. Overrides paths.work_dir.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the global seed.")
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--non-deterministic", action="store_true",
        help="Disable TensorFlow deterministic kernels (faster, not bit-reproducible).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pm25forecast",
        description="Multi-strategy deep ensembles for 48-hour urban PM2.5 forecasting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("preprocess", "Stage 1: clean, screen, impute, segment and scale every station."),
        ("train-base", "Stage 2: train the base sequence models."),
        ("train-ensembles", "Stage 3: fit the six ensembles (paper 4.2)."),
        ("evaluate", "Stage 4: score everything on the held-out test year."),
        ("ablation", "Stage 5: leave-one-member-out study (paper 5.5)."),
        ("run-all", "Run stages 1-5 end to end."),
        ("validate-config", "Parse and validate a config without touching any data."),
        ("info", "Print the resolved config and the environment fingerprint."),
    ):
        p = sub.add_parser(name, help=help_text, description=help_text)
        _common_arguments(p)
        if name in {"train-base", "train-ensembles", "run-all"}:
            p.add_argument(
                "--overwrite", action="store_true",
                help="Retrain models that already exist on disk.",
            )
        if name == "run-all":
            p.add_argument(
                "--skip-ablation", action="store_true",
                help="Skip the leave-one-out study, which refits every meta-learner once per "
                     "member and is by far the slowest stage.",
            )
        if name == "validate-config":
            p.add_argument(
                "--strict-paths", action="store_true",
                help="Also require that paths.raw_dir exists.",
            )
    return parser


def _overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {"paths": {}}
    if getattr(args, "raw_dir", None):
        overrides["paths"]["raw_dir"] = args.raw_dir
    if getattr(args, "work_dir", None):
        overrides["paths"]["work_dir"] = args.work_dir
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    if getattr(args, "non_deterministic", False):
        overrides["deterministic"] = False
    return overrides


def _load(args: argparse.Namespace, *, require_raw: bool = True) -> Config:
    cfg = load_config(
        args.config, overrides=_overrides_from_args(args), require_raw=require_raw
    )
    cfg.ensure_directories()
    configure_logging(args.log_level, log_file=cfg.work / "run.log")
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    dump_config(cfg, cfg.work / "resolved_config.yaml")
    return cfg


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging(args.log_level)

    try:
        if args.command == "validate-config":
            cfg = load_config(
                args.config,
                overrides=_overrides_from_args(args),
                require_raw=args.strict_paths,
            )
            logger.info("Configuration is valid.")
            logger.info("  raw_dir : %s", cfg.paths.raw_dir)
            logger.info("  work_dir: %s", cfg.paths.work_dir)
            logger.info("  target  : %s", cfg.data.target)
            logger.info(
                "  splits  : development %s..%s (%.0f%% train / %.0f%% val, %dh gap) | "
                "test %s..%s",
                cfg.split.development[0][:10], cfg.split.development[1][:10],
                cfg.split.base_train_ratio * 100, (1 - cfg.split.base_train_ratio) * 100,
                cfg.split.gap_hours,
                cfg.split.test[0][:10], cfg.split.test[1][:10],
            )
            logger.info("  members : %s", ", ".join(cfg.base_models.names))
            logger.info("  ensembles: %s", ", ".join(cfg.ensembles.names))
            if "xgb_meta" in cfg.ensembles.names:
                try:
                    import xgboost  # noqa: F401
                except ImportError:
                    logger.warning(
                        "  xgb_meta is configured but xgboost is not installed; the pipeline "
                        "will fall back to sklearn GradientBoosting. Run `pip install xgboost` "
                        "to reproduce the paper exactly."
                    )
            return 0

        if args.command == "info":
            cfg = load_config(
                args.config, overrides=_overrides_from_args(args), require_raw=False
            )
            print(json.dumps(
                {"config": cfg.to_dict(), "environment": environment_fingerprint()},
                indent=2, ensure_ascii=False, default=str,
            ))
            return 0

        cfg = _load(args)
        logger.info("Environment: %s", json.dumps(environment_fingerprint(), default=str))

        from .pipeline.ablation import run_ablation
        from .pipeline.evaluate import run_evaluation
        from .pipeline.preprocess import run_preprocessing
        from .pipeline.train_base import run_base_training
        from .pipeline.train_ensembles import run_ensemble_training

        if args.command == "preprocess":
            run_preprocessing(cfg)
        elif args.command == "train-base":
            run_base_training(cfg, overwrite=args.overwrite)
        elif args.command == "train-ensembles":
            run_ensemble_training(cfg, overwrite=args.overwrite)
        elif args.command == "evaluate":
            run_evaluation(cfg)
        elif args.command == "ablation":
            run_ablation(cfg)
        elif args.command == "run-all":
            run_preprocessing(cfg)
            run_base_training(cfg, overwrite=args.overwrite)
            run_ensemble_training(cfg, overwrite=args.overwrite)
            run_evaluation(cfg)
            if cfg.evaluation.run_ablation and not args.skip_ablation:
                run_ablation(cfg)
            section(logger, "PIPELINE COMPLETE")
            logger.info("Results: %s", cfg.results_dir)
        return 0

    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 2
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 3
    except KeyboardInterrupt:  # pragma: no cover
        logger.warning("Interrupted by user.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
