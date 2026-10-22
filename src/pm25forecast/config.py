"""Explicit scientific/data contract; defaults are not historical execution evidence."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import json
import pandas as pd

POLLUTANTS = ["PM 2.5 ug/m3", "O3 ppb", "CO ppm", "NO ppb", "NO2 ppb", "SO2 ppb"]
METEOROLOGY = ["tmpc", "relh", "alti", "vsby", "U", "V"]
CYCLICAL = ["hour_sin", "hour_cos"]
BASE_NAMES = ["bi_gru", "bi_lstm", "gru", "lstm"]
ENSEMBLE_NAMES = ["sa", "pwe", "mlp_meta", "xgb_meta", "h_mos", "sms"]


@dataclass
class Config:
    """Manuscript defaults plus explicit source-clock and availability contracts."""

    protocol: str = "manuscript_20261022"
    seed: int = 42
    raw_dir: str = ""
    output_dir: str = "./artifacts_manuscript"
    meteorology_file: str | None = None
    merged_meteorology_alignment_verified: bool = False
    datetime_column: str = "Date"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
    pollutant_timezone: str = "Asia/Tehran"
    meteorology_timezone: str = "UTC"
    ambiguous_local_time: str = "raise"
    nonexistent_local_time: str = "raise"
    visibility_unit: str = "m"
    pressure_unit: str = "hPa"
    wind_unit: str = "m/s"
    met_max_age_minutes: int = 60
    meteorology_reporting_delay_minutes: int = 0
    expected_stations: int = 13
    development_start: str = "2015-01-01 00:00:00"
    development_end: str = "2023-12-31 23:59:59"
    test_start: str = "2024-01-01 00:00:00"
    test_end: str = "2024-12-31 23:59:59"
    base_train_ratio: float = 0.7
    meta_train_ratio: float = 0.8
    gap_hours: int = 48
    input_hours: int = 48
    horizon_hours: int = 48
    train_stride: int = 1
    validation_stride: int = 48
    test_stride: int = 48
    min_target_hours_per_day: int = 18
    knn_neighbors: int = 5
    knn_weights: str = "uniform"
    z_threshold: float = 4.0
    z_jump: float = 9.0
    z_rolling_ratio: float = 2.0
    z_city_ratio: float = 2.0
    z_min_city_stations: int = 2
    epochs: int = 100
    batch_size: int = 512
    early_stopping_patience: int = 10
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 5
    min_learning_rate: float = 1e-6
    learning_rate: float = 0.001
    dropout: float = 0.3
    xgb_trees: int = 100
    xgb_depth: int = 6
    xgb_learning_rate: float = 0.1
    sms_trees: int = 100
    sms_depth: int = 10
    threshold: float = 35.0
    exclude_groups: list[str] = field(default_factory=list)
    grid: dict = field(default_factory=dict)
    grid_provenance: str = "Historical grid not supplied."

    @property
    def out(self):
        return Path(self.output_dir).resolve()

    @property
    def features(self):
        return (
            POLLUTANTS
            + ([] if "meteorology" in self.exclude_groups else METEOROLOGY)
            + ([] if "cyclical" in self.exclude_groups else CYCLICAL)
        )

    @property
    def context(self):
        return (
            ([] if "meteorology" in self.exclude_groups else METEOROLOGY)
            + ([] if "cyclical" in self.exclude_groups else CYCLICAL)
            + ([] if "offset" in self.exclude_groups else ["hour_offset"])
        )

    def timestamp(self, value):
        t = pd.Timestamp(value)
        return (
            t.tz_localize(
                self.pollutant_timezone, ambiguous="raise", nonexistent="raise"
            )
            if t.tzinfo is None
            else t
        ).tz_convert("UTC")

    @property
    def cut(self):
        start = self.timestamp(self.development_start)
        total = (
            int((self.timestamp(self.development_end) - start) / pd.Timedelta(hours=1))
            + 1
        )
        return start + pd.Timedelta(hours=int(total * self.base_train_ratio))

    @property
    def train_end(self):
        return self.cut - pd.Timedelta(hours=self.gap_hours + 1)

    def validate(self, require_data=False):
        if self.protocol != "manuscript_20261022":
            raise ValueError("Unknown protocol; historical outputs cannot be reused.")
        if (self.input_hours, self.horizon_hours) != (48, 48):
            raise ValueError(
                "Figure 2 TimeDistributed architecture requires the specified 48 -> 48 task."
            )
        if not 0 < self.base_train_ratio < 1 or not 0 < self.meta_train_ratio < 1:
            raise ValueError("Split ratios must lie strictly between 0 and 1.")
        if self.gap_hours < 48:
            raise ValueError("At least the manuscript 48-hour gap is required.")
        if self.test_stride < 48 or self.validation_stride < 48:
            raise ValueError("Verification and meta-training cycles must not overlap.")
        if not 1 <= self.min_target_hours_per_day <= 24:
            raise ValueError(
                "Daily target completeness must be between 1 and 24 hours."
            )
        if self.timestamp(self.development_end) >= self.timestamp(self.test_start):
            raise ValueError("Development/test periods overlap.")
        if self.timestamp(
            self.development_start
        ) >= self.train_end or self.cut >= self.timestamp(self.development_end):
            raise ValueError("Empty development split.")
        if self.timestamp(self.test_start) >= self.timestamp(self.test_end):
            raise ValueError("Empty test period.")
        if set(self.exclude_groups) - {"meteorology", "cyclical", "offset"}:
            raise ValueError("Unknown ablation group.")
        if (
            min(
                self.knn_neighbors,
                self.epochs,
                self.batch_size,
                self.train_stride,
                self.expected_stations,
                self.z_min_city_stations,
            )
            < 1
        ):
            raise ValueError("Counts and strides must be positive.")
        if not 0 < self.learning_rate or not 0 <= self.dropout < 1:
            raise ValueError("Invalid learning rate/dropout.")
        if not 0 < self.reduce_lr_factor < 1 or self.min_learning_rate <= 0:
            raise ValueError("Invalid learning-rate schedule.")
        if min(self.xgb_trees, self.xgb_depth, self.sms_trees, self.sms_depth) < 1:
            raise ValueError("Invalid tree configuration.")
        if self.met_max_age_minutes < 0 or self.meteorology_reporting_delay_minutes < 0:
            raise ValueError("Invalid weather availability policy.")
        if self.knn_weights not in {"uniform", "distance"}:
            raise ValueError("Unknown KNN weights.")
        if (
            self.visibility_unit not in {"m", "km", "mile"}
            or self.pressure_unit not in {"hPa", "inHg"}
            or self.wind_unit not in {"m/s", "knots"}
        ):
            raise ValueError("Declare supported source units.")
        if not self.meteorology_file and not self.merged_meteorology_alignment_verified:
            raise ValueError(
                "Merged meteorology time alignment is unverified. Provide separate UTC meteorology, or verify alignment explicitly. Do not guess."
            )
        if require_data:
            raw = Path(self.raw_dir).resolve()
            if not raw.is_dir():
                raise ValueError("Set raw_dir to the station CSV folder.")
            if self.out == raw or raw in self.out.parents:
                raise ValueError(
                    "Outputs must not be written inside the raw-data folder."
                )
            if self.meteorology_file and not Path(self.meteorology_file).is_file():
                raise ValueError("Meteorology file missing.")
        return self

    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


def load_config(path):
    """Read an explicit JSON configuration; reject unknown field names."""
    return Config(**json.loads(Path(path).read_text(encoding="utf-8-sig")))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()
