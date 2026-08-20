# pm25forecast — multi-strategy deep ensembles for 48-hour PM2.5 forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Reference implementation of a study that asks a specific question: **which way of
combining deep forecasters actually works for urban PM2.5?**

Four deep base learners per station, then **six ensembles spanning three distinct
strategies** — statistical averaging, non-linear meta-learning, and context-aware
selection — compared on identical windows, over a 48-hour horizon, across an
entire monitoring network.

Built for the Tehran network (hourly, 2015–2024), but nothing is Tehran-specific:
point `paths.raw_dir` at your own archive and every station in it is picked up
automatically.

---

## Start here

| If you want to… | Read |
|---|---|
| understand what the code does, stage by stage, in plain text | [`PROJECT_GUIDE.txt`](PROJECT_GUIDE.txt) |
| run it | the **Install** and **Quick start** sections below |
| find the function behind an equation in the paper | [`docs/PAPER_MAPPING.md`](docs/PAPER_MAPPING.md) |
| know why each stage is built the way it is | [`docs/METHODS.md`](docs/METHODS.md) |
| point it at your own data | [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) |

```
pm25-ensemble-forecasting/
├── PROJECT_GUIDE.txt    plain-text walkthrough of the whole pipeline
├── README.md            this file — commands and output layout
├── configs/             default.yaml is the published method
├── docs/                paper mapping, methods rationale, data schema
├── src/pm25forecast/    the package: one module per concern
│   ├── data/            loading, IQR screening, imputation, segments
│   ├── models/          the four base learners, the losses, the six ensembles
│   └── pipeline/        one module per stage, in run order
└── tests/               paper-fidelity, windowing, and end-to-end tests
```

---

## The three strategies

Each embodies a different theory of *why* combining models helps. The paper's
contribution is testing them against each other rather than asserting one.

| Strategy | Members | The claim being tested |
|---|---|---|
| **Statistical Averaging** | `SA`, `PWE` | Members are individually noisy but jointly unbiased, so a fixed linear blend cancels variance |
| **Non-Linear Meta-Learning** | `MLP-Meta`, `XGB-Meta` | Member error depends on atmospheric state in a way no linear blend can express |
| **Context-Aware Selection** | `H-MOS`, `SMS` | Members are *specialists* — rather than average them, work out which one to trust right now |

`SA` has no parameters, so it cannot overfit. That makes it the control every
other ensemble must beat to justify its machinery, and the pipeline reports it
alongside the rest for exactly that reason.

---

## Pipeline

```
raw CSVs
   │
   ├─ 1. PREPROCESS ── hourly grid → split FIRST → IQR screen (train-fitted)
   │                 → KNN impute (train-fitted) → features → segments
   │                 → min-max scaler (train-fitted, then verified)
   │
   ├─ 2. BASE LEARNERS  LSTM │ GRU  (64→32, inverse-weighted loss)
   │                    Bi-LSTM │ Bi-GRU  (2×32, exponential extreme loss)
   │
   ├─ 3. ENSEMBLES ──── fitted on VALIDATION predictions, six of them
   │
   ├─ 4. EVALUATE ───── 2024, read exactly once
   │                    daily means, Day 1 (1-24 h) vs Day 2 (25-48 h)
   │
   └─ 5. ABLATION ───── leave-one-member-out, every meta-learner refitted
```

**Split, and why it is shaped this way.** Development 2015–2023 → 70 % training,
a **48-hour gap**, 30 % validation. Test = 2024, touched once.

The 30 % validation block is large on purpose: it is not only an early-stopping
monitor, it is the *entire training set* for all six ensembles. And the 48-hour
gap is not decoration — with a 48-hour input window and a 48-hour horizon, a
window anchored at the last training hour occupies 95 further hours, so without a
purge the final training windows and the first validation windows would overlap
almost completely.

---

## Install

```bash
git clone https://github.com/sinadadras021/pm25-ensemble-forecasting.git
cd pm25-ensemble-forecasting

python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .              # or: pip install -e ".[cpu]" for tensorflow-cpu
pip install -e ".[dev]"       # to run the tests
```

Requires Python ≥ 3.10. A GPU is optional; the smoke config runs on a laptop CPU
in about a minute.

---

## Quick start

```bash
# 1. Check your configuration before spending any compute
pm25forecast validate-config --config configs/default.yaml --raw-dir /path/to/csvs --strict-paths

# 2. Prove the plumbing works on your archive (tiny models, ~1 minute)
pm25forecast run-all --config configs/quick_test.yaml --raw-dir /path/to/csvs

# 3. The real run
pm25forecast run-all --config configs/default.yaml --raw-dir /path/to/csvs
```

Stages also run individually — each reads the previous stage's artefacts from
disk, which is what you want while iterating:

```bash
pm25forecast preprocess      --config configs/default.yaml --raw-dir /path/to/csvs
pm25forecast train-base      --config configs/default.yaml
pm25forecast train-ensembles --config configs/default.yaml
pm25forecast evaluate        --config configs/default.yaml
pm25forecast ablation        --config configs/default.yaml
```

### Three ways to supply paths

```bash
pm25forecast run-all --raw-dir /data/pm25 --work-dir /scratch/run17   # CLI flags win
PM25_PATHS__RAW_DIR=/data/pm25 pm25forecast run-all                   # environment
# ...or edit paths.raw_dir in your YAML config
```

Precedence: **defaults → YAML → environment → CLI flags.** There are no
interactive prompts anywhere, by design: prompts cannot be scripted, cannot run
in CI, and cannot be recorded in a provenance file.

---

## Configuration

| file | purpose |
|---|---|
| `configs/default.yaml` | the published method, exactly |
| `configs/extended.yaml` | optional robustness settings, all **beyond the paper** |
| `configs/quick_test.yaml` | 1-epoch smoke run for CI and first contact with a new archive |

Every key is documented in `src/pm25forecast/config.py`, which *is* the schema.
The YAML only overrides fields that already exist there, and a typo raises rather
than being silently ignored:

```
ConfigError: Unknown configuration key 'ensembless.names'.
Valid keys at this level: ['base_models', 'data', 'ensembles', ...]
```

---

## Output layout

```
artifacts/
├── resolved_config.yaml            # exactly what ran, for provenance
├── run.log
├── 01_cleaned/                     # per-station parquet + scalers + IQR bounds
├── 02_base_models/<station>/       # *.keras, member_predictions.npz, history
├── 03_ensembles/<station>/         # sa.joblib ... sms.joblib, ensemble_meta.json
├── 04_results/
│   ├── station_quality_report.csv     # who was admitted, and why not
│   ├── metrics_all_stations.csv
│   ├── metrics_city_mean.csv
│   ├── metrics_by_strategy.csv        # the comparison the paper is making
│   ├── ablation_all_stations.csv
│   └── predictions_<station>.parquet  # redraw any figure without retraining
└── 05_figures/
```

`ensemble_meta.json` carries the parts of each ensemble that are worth reading on
their own: PWE's optimised weights, the SMS selector's accuracy and its oracle
gain, H-MOS's 24 hourly (α, β) pairs, and XGB-Meta's top features.

---

## What the code refuses to do

Enforced at runtime, not left to reviewer trust:

- **Fit any transform outside the training block.** `SplitScaler` and
  `TrainFittedKNNImputer` record the timestamps they saw and re-derive the claim
  on demand; `IQRScreen` is a two-phase estimator whose `fit` takes only training
  data. The config *rejects* `fit_on: development` for all three.
- **Clip an outlier into range.** Excluded values become `NaN` and are imputed.
  Clipping the target would truncate exactly the pollution episodes the model
  exists to predict.
- **Screen validation or test labels.** Whatever the config says, a ground truth
  that has been altered is no longer a ground truth.
- **Fit an ensemble on in-sample base predictions.** All six are fitted on the
  validation block, and a guard re-derives the provenance of the arrays being
  used.
- **Let a window cross a data gap or a split boundary.** Segments break on gaps in
  the timestamp index as well as on `NaN`; window intervals are reconstructed and
  compared across splits.
- **Score overlapping test windows.** `windows.stride.test` must be ≥ the horizon,
  or the config is rejected — otherwise the same observation is counted many times
  and the "sample size" is fiction.

---

## Results you should expect to see reported

For every station and every model, at three levels:

- **Hourly** — did the model track the diurnal cycle, or reach a good daily mean
  by averaging two opposite errors?
- **Day 1 (1–24 h) and Day 2 (25–48 h) daily means** — the paper's headline, and
  where the decay of predictability becomes visible: RMSE, MAE, R (Pearson), IOA,
  Unc%.
- **By strategy** — the three families compared directly.

R and IOA are reported together on purpose. A forecast that tracks every rise and
fall but sits 20 µg/m³ too high has R ≈ 1 and a poor IOA; seeing both makes the
difference between "wrong shape" and "right shape, wrong level" immediately
legible.

If no ensemble beats the best single base learner, the log says so.

---

## Testing

```bash
pytest                     # fast unit + paper-fidelity tests
pytest -m slow             # + the end-to-end run on synthetic stations
pytest --cov=pm25forecast  # with coverage
```

`tests/test_paper_fidelity.py` pins the implementation to the manuscript: each
test names the section or equation it protects, so if a default changes later,
one of them fails and says which published claim it just broke.

The synthetic stations in `tests/conftest.py` are generated from the mechanisms
that actually drive urban PM2.5 — boundary-layer collapse, traffic peaks, wind
ventilation, winter inversions — then deliberately damaged with outages,
instrument spikes and duplicated timestamps.

---

## Data

**No data is distributed with this repository.** The observations used in the
associated study belong to the Tehran Air Quality Control Company (pollutants)
and the Iran Meteorological Organization (meteorology), and are available from
them on request.

[`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) specifies exactly what the code
expects, so the pipeline can be pointed at any hourly monitoring archive with the
same variables.

---

## Documentation

- [`docs/PAPER_MAPPING.md`](docs/PAPER_MAPPING.md) — every equation and table of
  the paper mapped to the function that implements it
- [`docs/METHODS.md`](docs/METHODS.md) — the rationale for each stage and an
  explicit list of assumptions and limitations
- [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) — input format, units, adaptation

---

## Citation

If this code contributes to your work, please cite it via
[`CITATION.cff`](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
