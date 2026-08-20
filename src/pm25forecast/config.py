"""Typed, validated configuration for the PM2.5 multi-ensemble pipeline.

Every default is the value stated in the paper; section numbers in the docstrings
refer to the manuscript, and ``docs/PAPER_MAPPING.md`` maps each equation and
table to the function that implements it. A reader with the paper open should
find the corresponding knob here without guessing.

Anything **not** described in the paper is either absent or exposed as a flag
that is **off by default** and labelled "beyond the paper".
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

__all__ = [
    "ConfigError",
    "PathsConfig",
    "DataConfig",
    "OutlierConfig",
    "ImputationConfig",
    "FeatureConfig",
    "SplitConfig",
    "WindowConfig",
    "ScalingConfig",
    "BaseModelConfig",
    "EnsembleConfig",
    "EvaluationConfig",
    "Config",
    "load_config",
    "dump_config",
]

_ENV_PREFIX = "PM25_"
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")


class ConfigError(ValueError):
    """A configuration file is structurally or semantically invalid."""


# --------------------------------------------------------------------------- #
@dataclass
class PathsConfig:
    """Filesystem locations. ``raw_dir`` is the only value you must supply."""

    raw_dir: str = "<SET ME: directory containing one CSV per station>"
    work_dir: str = "./artifacts"

    def resolve(self) -> "PathsConfig":
        return PathsConfig(
            raw_dir=str(Path(os.path.expandvars(os.path.expanduser(self.raw_dir))).resolve()),
            work_dir=str(Path(os.path.expandvars(os.path.expanduser(self.work_dir))).resolve()),
        )

    def validate(self, *, require_raw: bool = True) -> None:
        if "<SET ME" in self.raw_dir:
            raise ConfigError(
                "paths.raw_dir is still the placeholder. Set it in your YAML config, pass "
                "--raw-dir on the command line, or export PM25_PATHS__RAW_DIR."
            )
        if require_raw and not Path(self.raw_dir).is_dir():
            raise ConfigError(f"paths.raw_dir does not exist or is not a directory: {self.raw_dir}")


@dataclass
class DataConfig:
    """Raw input contract (paper §2.1)."""

    station_glob: str = "*.csv"
    datetime_column: str = "Date"
    datetime_formats: List[str] = field(
        default_factory=lambda: ["%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]
    )
    target: str = "PM 2.5 ug/m3"
    #: Columns removed before modelling.
    #:
    #: ``PM 10`` is strongly collinear with PM2.5 and would let the model read
    #: most of its own answer off a co-measured channel; ``NOx`` is NO + NO2 by
    #: construction; ``feel`` (apparent temperature) is a deterministic function
    #: of temperature, humidity and wind, all already present.
    drop_columns: List[str] = field(
        default_factory=lambda: ["PM 10 ug/m3", "NOx ppb", "feel"]
    )
    #: Analysis window, clipped from every station so all share one grid.
    global_start: str = "2015-01-01 00:00"
    global_end: str = "2024-12-30 23:00"
    #: Minimum fraction of observed target values for a station to be admitted.
    #: **Beyond the paper**, which does not state a station-admission rule; set
    #: to 0 to admit every station regardless of coverage.
    min_target_completeness: float = 0.0

    def validate(self) -> None:
        if not self.target:
            raise ConfigError("data.target must be a non-empty column name.")
        if self.target in self.drop_columns:
            raise ConfigError(f"data.target ({self.target!r}) also appears in data.drop_columns.")
        for name in ("global_start", "global_end"):
            _require_timestamp(getattr(self, name), f"data.{name}")
        if self.global_start >= self.global_end:
            raise ConfigError("data.global_start must be strictly before data.global_end.")
        _require_unit_interval(self.min_target_completeness, "data.min_target_completeness")


@dataclass
class OutlierConfig:
    r"""Interquartile-range outlier removal (paper §2.2).

    Bounds are :math:`Q_1 - k \cdot \mathrm{IQR}` and :math:`Q_3 + k \cdot
    \mathrm{IQR}` with :math:`k = 1.5`, and values outside them are **excluded**
    from the dataset.

    Two properties are enforced here that a naive implementation gets wrong, and
    both change the reported numbers:

    *   **Quantiles are computed on the training block only.** Deriving them from
        the whole record — or, worse, separately from the test year — lets the
        test distribution shape its own preprocessing.
    *   **Excluded values become NaN and are then imputed; they are not clipped
        to the boundary.** Clipping converts a detectable fault into a
        plausible-looking observation, and clipping the *target* truncates the
        very pollution episodes the model exists to predict.
    """

    enabled: bool = True
    #: Multiplier on the IQR (paper §2.2).
    k: float = 1.5
    fit_on: str = "train"
    #: Columns screened. Empty list means "every numeric column".
    columns: List[str] = field(default_factory=list)
    #: Screen the target as well as the features.
    #:
    #: The paper applies the rule to "all input variables". Screening the target
    #: too is defensible for the *training* block, but must never touch
    #: validation or test labels, which is enforced regardless of this setting.
    include_target: bool = True

    def validate(self) -> None:
        if self.k <= 0:
            raise ConfigError("outliers.k must be > 0.")
        if self.fit_on != "train":
            raise ConfigError(
                "outliers.fit_on must be 'train'. Deriving IQR bounds from any block that "
                "includes validation or test data is leakage."
            )


@dataclass
class ImputationConfig:
    """k-nearest-neighbours imputation (paper §2.2).

    "To avoid information leakage, the imputation model was trained on the
    training data and then applied to the validation and test sets."
    """

    n_neighbors: int = 5
    weights: str = "uniform"
    #: Interpolate gaps at most this many hours long before KNN. **Beyond the
    #: paper**; 0 disables it and hands every gap to the KNN model.
    short_gap_hours: int = 0
    fit_on: str = "train"
    #: Whether the *label* is taken from the imputer as well as the inputs.
    #:
    #: The encoder always reads a gap-filled copy of the target history: at issue
    #: time an operational system with a one-hour sensor dropout does fill it, and
    #: refusing to would discard the whole 96-hour window over a single missing
    #: input.
    #:
    #: ``True`` matches the paper, which applies KNN to "missing values" and then
    #: to the validation and test sets without excepting the target.
    #: ``False`` keeps the label as observations only; the losses and metrics mask
    #: the gaps, and ``windows.min_label_completeness`` bounds how incomplete a
    #: retained window may be. Stricter, and **beyond the paper**.
    impute_target: bool = True

    def validate(self) -> None:
        if self.n_neighbors < 1:
            raise ConfigError("imputation.n_neighbors must be >= 1.")
        if self.weights not in {"uniform", "distance"}:
            raise ConfigError("imputation.weights must be 'uniform' or 'distance'.")
        if self.fit_on != "train":
            raise ConfigError(
                "imputation.fit_on must be 'train' -- the paper is explicit that the imputer "
                "is fitted on training data only."
            )
        if self.short_gap_hours < 0:
            raise ConfigError("imputation.short_gap_hours must be >= 0.")


@dataclass
class FeatureConfig:
    """Feature construction (paper §2.2)."""

    #: Wind speed/direction replaced by U and V components (Eqs. 1-2). A no-op
    #: when the archive already stores U and V.
    wind_components: bool = True
    wind_speed_column: str = "wind_speed"
    wind_direction_column: str = "wind_direction"
    #: Cyclical hour encodings (Eqs. 3-4).
    cyclical: List[str] = field(default_factory=lambda: ["hour"])
    #: Supply ``Hour_offset`` (0 .. horizon-1) to the meta-learners, so they can
    #: model how member reliability decays with lead time (paper §4.2).
    hour_offset_feature: bool = True
    #: Additional day-of-year encoding. **Beyond the paper.**
    add_day_of_year: bool = False

    def validate(self) -> None:
        allowed = {"hour", "dayofweek", "doy"}
        unknown = sorted(set(self.cyclical) - allowed)
        if unknown:
            raise ConfigError(f"features.cyclical contains unsupported entries: {unknown}")


@dataclass
class SplitConfig:
    """Chronological split (paper §2.2).

    ::

        |<------------- development 2015-2023 ------------->|<-- test 2024 -->|
        |<--- 70% base training --->| gap |<--- 30% val --->|

    The 48-hour gap matters more here than it looks. With a 48-hour input window
    and a 48-hour horizon, a window anchored at the last training hour reaches
    95 hours into the future; without a purge band the final training windows and
    the first validation windows would overlap almost entirely, and early
    stopping would be reading its own training data.
    """

    development: Tuple[str, str] = ("2015-01-01 00:00", "2023-12-30 23:00")
    test: Tuple[str, str] = ("2024-01-01 00:00", "2024-12-30 23:00")
    #: Fraction of development used for base-model training (paper §2.2).
    base_train_ratio: float = 0.70
    #: Hours discarded between the training and validation blocks (paper §2.2).
    gap_hours: int = 48
    #: Train/validation split inside each meta-learner's dataset (paper §2.2).
    meta_train_ratio: float = 0.80

    def validate(self) -> None:
        import pandas as pd

        for name in ("development", "test"):
            span = getattr(self, name)
            if len(span) != 2:
                raise ConfigError(f"split.{name} must be a [start, end] pair.")
            _require_timestamp(span[0], f"split.{name}[0]")
            _require_timestamp(span[1], f"split.{name}[1]")
            if pd.Timestamp(span[0]) >= pd.Timestamp(span[1]):
                raise ConfigError(f"split.{name}: start must be strictly before end.")
        if pd.Timestamp(self.development[1]) >= pd.Timestamp(self.test[0]):
            raise ConfigError(
                f"split.development ends at {self.development[1]}, which is not before "
                f"split.test start {self.test[0]}. The test year must lie strictly in the "
                f"future of everything the model has seen."
            )
        for name in ("base_train_ratio", "meta_train_ratio"):
            _require_unit_interval(
                getattr(self, name), f"split.{name}", closed_left=False, closed_right=False
            )
        if self.gap_hours < 0:
            raise ConfigError("split.gap_hours must be >= 0.")


@dataclass
class WindowConfig:
    """Sliding-window geometry (paper Table 3)."""

    input_hours: int = 48
    horizon_hours: int = 48
    #: Stride per split. Stride 1 in training maximises samples; the test stride
    #: equals the horizon so scored windows never overlap and each observation is
    #: counted once.
    stride: Dict[str, int] = field(
        default_factory=lambda: {"train": 1, "val": 1, "test": 48}
    )
    min_windows: Dict[str, int] = field(
        default_factory=lambda: {"train": 500, "val": 100, "test": 20}
    )
    #: Minimum fraction of the horizon that must be observed for a window to be
    #: kept; missing label hours are masked by the losses and the metrics.
    min_label_completeness: float = 0.75
    #: Keep only windows whose first forecast hour falls on this clock hour, so
    #: each window is exactly two calendar days -- which is what makes the
    #: Day 1 / Day 2 evaluation of paper §4.4 well defined.
    daily_anchor_hour: Optional[int] = 0
    #: Longest gap tolerated inside a window before the series is split.
    max_gap_hours: int = 0

    def validate(self) -> None:
        if self.input_hours < 1 or self.horizon_hours < 1:
            raise ConfigError("windows.input_hours and windows.horizon_hours must be >= 1.")
        for split_name, value in self.stride.items():
            if value < 1:
                raise ConfigError(f"windows.stride['{split_name}'] must be >= 1.")
        if self.stride.get("test", self.horizon_hours) < self.horizon_hours:
            raise ConfigError(
                "windows.stride['test'] must be >= windows.horizon_hours, otherwise the same "
                "observation is scored several times and the test metrics are not independent "
                "samples."
            )
        _require_unit_interval(
            self.min_label_completeness, "windows.min_label_completeness", closed_left=False
        )
        if self.daily_anchor_hour is not None and not 0 <= self.daily_anchor_hour <= 23:
            raise ConfigError("windows.daily_anchor_hour must be null or in 0..23.")


@dataclass
class ScalingConfig:
    """Min-max normalisation to [0, 1] (paper §2.2), fitted on training data."""

    method: str = "minmax"
    fit_on: str = "train"
    clip_scaled: Optional[Tuple[float, float]] = None

    def validate(self) -> None:
        if self.method not in {"minmax", "robust", "standard"}:
            raise ConfigError("scaling.method must be one of minmax|robust|standard.")
        if self.fit_on != "train":
            raise ConfigError(
                "scaling.fit_on must be 'train' -- the paper states the scaling parameters "
                "were derived exclusively from the training set."
            )


@dataclass
class BaseModelConfig:
    """The four base learners (paper §4.1, Table 2).

    Two families with deliberately different objectives:

    ================  =====================  ===============================
    family            structure              loss
    ================  =====================  ===============================
    ``lstm``/``gru``  64 -> 32 (ReLU)        inverse-weighted (IWL)
    ``bi_*``          2 x 32 (ReLU)          exponential extreme (EEL)
    ================  =====================  ===============================
    """

    names: List[str] = field(default_factory=lambda: ["lstm", "gru", "bi_lstm", "bi_gru"])
    rnn_units: Tuple[int, int] = (64, 32)
    bi_units: Tuple[int, int] = (32, 32)
    dropout: float = 0.20
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-5
    #: Weighting intensity alpha of the IWL (paper Eq. 19).
    iwl_alpha: float = 10.0
    #: Power p on the residual in the EEL (paper Eq. 20).
    eel_power: float = 2.0

    def validate(self) -> None:
        from .models.base import AVAILABLE_BASE_MODELS

        unknown = sorted(set(self.names) - set(AVAILABLE_BASE_MODELS))
        if unknown:
            raise ConfigError(
                f"base_models.names contains unknown architectures: {unknown}. "
                f"Available: {sorted(AVAILABLE_BASE_MODELS)}"
            )
        if not self.names:
            raise ConfigError("base_models.names must list at least one architecture.")
        if self.iwl_alpha < 0:
            raise ConfigError("base_models.iwl_alpha must be >= 0.")
        if self.eel_power <= 0:
            raise ConfigError("base_models.eel_power must be > 0.")
        _require_unit_interval(self.dropout, "base_models.dropout", closed_right=False)


@dataclass
class EnsembleConfig:
    """The six ensembles, in three strategies (paper §4.2, Table 2)."""

    names: List[str] = field(
        default_factory=lambda: ["sa", "pwe", "mlp_meta", "xgb_meta", "h_mos", "sms"]
    )
    #: MLP-Meta hidden layers (Table 2: 32 -> 16 -> 8, ReLU).
    mlp_hidden: Tuple[int, int, int] = (32, 16, 8)
    mlp_dropout: float = 0.10
    mlp_learning_rate: float = 1e-3
    mlp_epochs: int = 100
    mlp_batch_size: int = 256
    mlp_early_stopping_patience: int = 10
    #: XGB-Meta (Table 2: 100 trees, depth 6, learning rate 0.1).
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 1.0
    #: SMS random-forest classifier (Table 2: 100 trees, depth 6, Gini).
    sms_n_estimators: int = 100
    sms_max_depth: int = 6
    #: PWE optimiser (Table 2: SLSQP, convex weights summing to 1).
    pwe_method: str = "SLSQP"
    pwe_max_iter: int = 1000

    def validate(self) -> None:
        from .models.ensembles import AVAILABLE_ENSEMBLES

        unknown = sorted(set(self.names) - set(AVAILABLE_ENSEMBLES))
        if unknown:
            raise ConfigError(
                f"ensembles.names contains unknown entries: {unknown}. "
                f"Available: {sorted(AVAILABLE_ENSEMBLES)}"
            )
        _require_unit_interval(self.mlp_dropout, "ensembles.mlp_dropout", closed_right=False)


@dataclass
class EvaluationConfig:
    """Scoring (paper §4.4, Eqs. 14-18).

    The 48-hour forecasts are aggregated into **daily means** and reported for two
    sequential 24-hour windows, Day 1 and Day 2, so that skill is measured against
    the standardised daily benchmarks a public-health office actually uses -- and
    so that the decay of predictability with lead time is visible rather than
    averaged away.
    """

    #: Minimum observed hours for a day's mean to be defined.
    min_hours_per_day: int = 18
    #: Alert level for exceedance statistics. **Beyond the paper**, which reports
    #: only continuous metrics; ``null`` disables event scoring.
    peak_threshold_ugm3: Optional[float] = 35.0
    prediction_clip: Optional[Tuple[float, float]] = (0.0, 500.0)
    #: Reference forecasts. **Beyond the paper**; empty reproduces it exactly.
    baselines: List[str] = field(default_factory=list)
    bootstrap_samples: int = 0
    bootstrap_alpha: float = 0.05
    #: Leave-one-member-out ablation over the meta-learners (paper §5.5).
    run_ablation: bool = True

    def validate(self) -> None:
        from .baselines import AVAILABLE_BASELINES

        unknown = sorted(set(self.baselines) - set(AVAILABLE_BASELINES))
        if unknown:
            raise ConfigError(
                f"evaluation.baselines contains unknown entries: {unknown}. "
                f"Available: {sorted(AVAILABLE_BASELINES)}"
            )
        if self.min_hours_per_day < 1 or self.min_hours_per_day > 24:
            raise ConfigError("evaluation.min_hours_per_day must be in 1..24.")
        if self.prediction_clip is not None and self.prediction_clip[0] >= self.prediction_clip[1]:
            raise ConfigError("evaluation.prediction_clip must be [low, high] with low < high.")


@dataclass
class Config:
    """Root configuration."""

    project: str = "pm25-ensemble-forecasting"
    seed: int = 42
    deterministic: bool = True
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    outliers: OutlierConfig = field(default_factory=OutlierConfig)
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    base_models: BaseModelConfig = field(default_factory=BaseModelConfig)
    ensembles: EnsembleConfig = field(default_factory=EnsembleConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def work(self) -> Path:
        return Path(self.paths.work_dir)

    @property
    def cleaned_dir(self) -> Path:
        return self.work / "01_cleaned"

    @property
    def artifacts_dir(self) -> Path:
        return self.work / "02_base_models"

    @property
    def ensemble_dir(self) -> Path:
        return self.work / "03_ensembles"

    @property
    def results_dir(self) -> Path:
        return self.work / "04_results"

    @property
    def figures_dir(self) -> Path:
        return self.work / "05_figures"

    def ensure_directories(self) -> None:
        for path in (
            self.work, self.cleaned_dir, self.artifacts_dir,
            self.ensemble_dir, self.results_dir, self.figures_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def validate(self, *, require_raw: bool = True) -> "Config":
        self.paths.validate(require_raw=require_raw)
        self.data.validate()
        self.outliers.validate()
        self.imputation.validate()
        self.features.validate()
        self.split.validate()
        self.windows.validate()
        self.scaling.validate()
        self.base_models.validate()
        self.ensembles.validate()
        self.evaluation.validate()
        self._validate_cross_section()
        return self

    def _validate_cross_section(self) -> None:
        import pandas as pd

        if self.windows.daily_anchor_hour is not None and self.windows.horizon_hours % 24:
            import warnings

            warnings.warn(
                f"windows.daily_anchor_hour is set but the horizon ({self.windows.horizon_hours}h) "
                "is not a whole number of days, so the Day 1 / Day 2 split of paper 4.4 is not "
                "well defined.",
                RuntimeWarning,
                stacklevel=2,
            )
        g_start, g_end = pd.Timestamp(self.data.global_start), pd.Timestamp(self.data.global_end)
        for name in ("development", "test"):
            start, end = getattr(self.split, name)
            if pd.Timestamp(start) < g_start or pd.Timestamp(end) > g_end:
                raise ConfigError(
                    f"split.{name} = [{start}, {end}] falls outside the analysis window "
                    f"[{self.data.global_start}, {self.data.global_end}]."
                )
        if self.split.gap_hours < self.windows.horizon_hours:
            import warnings

            warnings.warn(
                f"split.gap_hours ({self.split.gap_hours}) is smaller than the forecast horizon "
                f"({self.windows.horizon_hours}). Training and validation windows can then "
                f"overlap in time, and early stopping partly reads its own training data.",
                RuntimeWarning,
                stacklevel=2,
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
def _require_timestamp(value: str, key: str) -> None:
    if not isinstance(value, str) or not _TIMESTAMP_RE.match(value.strip()):
        raise ConfigError(
            f"{key} must be an ISO-like timestamp string such as '2024-01-01 00:00'; got {value!r}"
        )


def _require_unit_interval(
    value: float, key: str, *, closed_left: bool = True, closed_right: bool = True
) -> None:
    low_ok = value >= 0.0 if closed_left else value > 0.0
    high_ok = value <= 1.0 if closed_right else value < 1.0
    if not (low_ok and high_ok):
        raise ConfigError(f"{key} must lie in the unit interval; got {value!r}")


def _coerce(value: Any, target_type: Any) -> Any:
    origin = getattr(target_type, "__origin__", None)
    if origin is tuple and isinstance(value, (list, tuple)):
        return tuple(value)
    if target_type is float and isinstance(value, (int, str)):
        return float(value)
    if target_type is int and isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return value


def _merge_into(instance: Any, overrides: Mapping[str, Any], path: str = "") -> Any:
    if not is_dataclass(instance):
        return overrides
    known = {f.name: f for f in fields(instance)}
    for key, value in overrides.items():
        dotted = f"{path}.{key}".lstrip(".")
        if key not in known:
            raise ConfigError(
                f"Unknown configuration key '{dotted}'. Valid keys at this level: {sorted(known)}"
            )
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _merge_into(current, value, dotted)
        else:
            setattr(instance, key, _coerce(value, known[key].type))
    return instance


def _env_overrides() -> Dict[str, Any]:
    """Translate ``PM25_SECTION__KEY=value`` variables into a nested dict."""
    nested: Dict[str, Any] = {}
    for raw_key, raw_value in os.environ.items():
        if not raw_key.startswith(_ENV_PREFIX):
            continue
        parts = [p.lower() for p in raw_key[len(_ENV_PREFIX) :].split("__") if p]
        if not parts:
            continue
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested


def _drop_none(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            nested = _drop_none(value)
            if nested:
                out[key] = nested
        elif value is not None:
            out[key] = value
    return out


def load_config(
    path: Optional[str | os.PathLike[str]] = None,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
    require_raw: bool = True,
    validate: bool = True,
) -> Config:
    """Build a validated :class:`Config`.

    Precedence, lowest to highest: dataclass defaults, YAML file, ``PM25_*``
    environment variables, explicit ``overrides`` (where CLI flags land).
    Unknown keys raise rather than being ignored.
    """
    cfg = Config()

    if path is not None:
        file_path = Path(path)
        if not file_path.is_file():
            raise ConfigError(f"Config file not found: {file_path}")
        with file_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, Mapping):
            raise ConfigError(f"{file_path} must contain a YAML mapping at the top level.")
        _merge_into(cfg, payload)

    env = _env_overrides()
    if env:
        _merge_into(cfg, env)
    if overrides:
        _merge_into(cfg, _drop_none(overrides))

    cfg.paths = cfg.paths.resolve()
    if validate:
        cfg.validate(require_raw=require_raw)
    return cfg


def dump_config(cfg: Config, destination: str | os.PathLike[str]) -> Path:
    """Write the fully-resolved configuration next to the results it produced."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False, allow_unicode=True)
    return target
