# Paper → code map

Every numbered section, equation and table of the manuscript points to the
function that implements it. Open the paper beside this file and any claim in the
text can be checked against running code in one step.

---

## 2. Study Area and Data

| Paper | Implementation | What it does |
|---|---|---|
| §2.1 Study area and data collection | [`data/loading.py`](../src/pm25forecast/data/loading.py) · `discover_stations`, `read_station_csv` | Stations are discovered from a directory, never hard-coded. Add a CSV and it joins the study. |
| §2.2 Wind transformation, **Eqs. 1–2** | [`features.py`](../src/pm25forecast/features.py) · `wind_to_components` | `U = −WS·sin(π/180·WD)`, `V = −WS·cos(π/180·WD)`. A no-op when the archive already stores U/V. |
| §2.2 Cyclical time, **Eqs. 3–4** | `features.py` · `add_cyclical_time_features` | `Hour_sin`, `Hour_cos`. |
| §2.2 `Hour_offset` (0–47) | `features.py` · `hour_offset_grid` | Supplied to the meta-learners so they can model lead-time decay. |
| §2.2 IQR outlier removal (k = 1.5) | [`data/quality.py`](../src/pm25forecast/data/quality.py) · `IQRScreen` | Bounds fitted on training data; flagged values are **voided and then imputed**, never clipped to the boundary. |
| §2.2 Data splitting | [`config.py`](../src/pm25forecast/config.py) · `SplitConfig` · [`pipeline/preprocess.py`](../src/pm25forecast/pipeline/preprocess.py) · `split_blocks` | Development 2015–2023 → 70 / 30 with a **48-hour gap**; test = 2024. Meta-learners split 80 / 20. |
| §2.2 KNN imputation | [`data/imputation.py`](../src/pm25forecast/data/imputation.py) · `TrainFittedKNNImputer` | k = 5, fitted on training data, applied to validation and test. `assert_fitted_on` re-derives the claim from recorded state. |
| §2.2 Min–max normalisation | [`scaling.py`](../src/pm25forecast/scaling.py) · `SplitScaler` | Scaled to `[0, 1]`, parameters from the training set only. |

Values excluded by the IQR screen become `NaN` and are handed to the imputer.
Clipping them to the boundary instead would convert a detectable fault into a
plausible observation, and applied to the target it would truncate exactly the
pollution episodes the model exists to predict.

---

## 4. Methodology

### 4.1 Base learners (Table 2)

| Paper | Implementation |
|---|---|
| LSTM, GRU — 64 → 32 (ReLU) | [`models/base.py`](../src/pm25forecast/models/base.py) · `_lstm`, `_gru` (`rnn_units: [64, 32]`) |
| Bi-LSTM, Bi-GRU — 2 × 32 (ReLU) | `models/base.py` · `_bi_lstm`, `_bi_gru` (`bi_units: [32, 32]`) |
| Adam, lr 0.001, 100 epochs, batch 32, patience 10 | `configs/default.yaml` → `base_models.*` |
| **Eq. 19** — Inverse-Weighted Loss (IWL) | [`models/losses.py`](../src/pm25forecast/models/losses.py) · `InverseWeightedLoss` (α = 10) |
| **Eq. 20** — Exponential Extreme Loss (EEL) | `models/losses.py` · `ExponentialExtremeLoss` (p = 2) |
| Table 2 loss assignment: RNNs → IWL, Bi-RNNs → EEL | `models/losses.py` · `DEFAULT_LOSS_BY_MODEL` |

Both custom losses normalise by the **count of observed labels**, not by the sum
of the weights. Dividing a weighted loss by its own weight sum would cancel the
weighting exactly — `w·ℓ / w = ℓ` — leaving a plain MSE wearing the name of an
extreme-value loss. Pinned by `tests/test_paper_fidelity.py`.

### 4.2 Ensembles (Table 2)

| Strategy | Model | Paper | Implementation |
|---|---|---|---|
| 4.2.1 Statistical Averaging | **SA** | Eq. 6 | [`models/ensembles.py`](../src/pm25forecast/models/ensembles.py) · `SimpleAveraging` |
| | **PWE** | Eqs. 7–8 | `PerformanceWeightedEnsemble` — SLSQP, `Σw = 1`, `0 ≤ wᵢ ≤ 1` |
| 4.2.2 Non-Linear Meta-Learning | **MLP-Meta** | Eq. 9 | `MLPMetaLearner` — 32 → 16 → 8, ReLU, Adam, MSE |
| | **XGB-Meta** | Eq. 10 | `XGBoostMetaLearner` — 100 trees, depth 6, lr 0.1 |
| 4.2.3 Context-Aware Selection | **H-MOS** | Eq. 11 | `HourlyMOS` — 24 hour-specific OLS correctors |
| | **SMS** | Eqs. 12–13 | `SmartModelSelector` — RF classifier, 100 trees, depth 6, Gini |

Meta-input space (§4.2): base predictions + meteorology + cyclical time +
`Hour_offset`, assembled in
[`pipeline/train_base.py`](../src/pm25forecast/pipeline/train_base.py) ·
`_forecast_context`.

All six ensembles are fitted on **validation** predictions. `SA` has no
parameters and therefore cannot overfit, which makes it the control every other
ensemble must beat to justify its machinery; the pipeline reports it alongside
the rest for that reason, and the log states plainly when no ensemble beats the
best single base learner.

### 4.3–4.4 Setup and metrics

| Paper | Implementation |
|---|---|
| Table 3 — history 48 h, horizon 48 h, Adam, 100 epochs, batch 32, patience 10 | `configs/default.yaml` → `windows.*`, `base_models.*` |
| §4.4 aggregation to **daily means**, Day 1 / Day 2 | [`metrics.py`](../src/pm25forecast/metrics.py) · `daily_means`, `split_days` |
| **Eq. 14** RMSE · **Eq. 15** MAE · **Eq. 16** R (Pearson) · **Eq. 17** IOA · **Eq. 18** Unc% | `metrics.py` · `continuous_metrics` |
| §4.4 Temporal stability (Day 1 vs Day 2) | `pipeline/evaluate.py` → `day1_*` / `day2_*` columns |
| §4.4 Loss optimisation analysis | `models/losses.py`, and the per-member rows in `metrics_all_stations.csv` |

R and IOA are reported together on purpose. A forecast that tracks every rise and
fall but sits 20 µg/m³ too high has R ≈ 1 and a poor IOA; reporting both makes
the difference between "wrong shape" and "right shape, wrong level" immediately
legible.

---

## 5. Results

| Paper | Implementation |
|---|---|
| §5.1 Ensemble superiority over base learners | `04_results/metrics_city_mean.csv`, `metrics_by_strategy.csv` |
| §5.3 Spatial distribution | `04_results/metrics_all_stations.csv` |
| §5.5 Ablation experiment | [`pipeline/ablation.py`](../src/pm25forecast/pipeline/ablation.py) · `run_ablation` |
| §5.6 Limitations | [`docs/METHODS.md`](METHODS.md) §7 |

The ablation refits every meta-learner after removing each member in turn, using
a chronological split throughout. Random cross-validation on an hourly series
would place near-duplicate neighbours on both sides of every fold, and the
resulting number would measure interpolation rather than forecasting.

---

## Five rules the code enforces at runtime

These are the properties on which the reported numbers depend, so they are
asserted while the pipeline runs rather than left to the reader's trust.

| Rule | Where |
|---|---|
| **No transform is fitted outside the training block.** `SplitScaler` and `TrainFittedKNNImputer` record the timestamps they saw and re-derive the claim on demand; `IQRScreen` is a two-phase estimator whose `fit` accepts training data only. The config rejects `fit_on: development` for all three. | `scaling.py`, `data/imputation.py`, `data/quality.py` |
| **Validation and test labels are never screened.** A ground truth that has been altered is no longer a ground truth, whatever the config says. | `data/quality.py` · `IQRScreen.transform` |
| **No ensemble is fitted on in-sample base predictions.** All six are fitted on the validation block, and a guard re-derives the provenance of the arrays being used. | `pipeline/train_ensembles.py` |
| **A window never spans a data gap or a split boundary.** Segments break on discontinuities in the timestamp index as well as on `NaN`; window intervals are reconstructed and compared across splits. | `data/segments.py`, `windowing.py`, `validation.py` |
| **Scored test windows never overlap.** `windows.stride.test` must be at least the horizon, or the config is rejected — otherwise the same observation is counted many times and the reported sample size is fiction. | `config.py` · `WindowConfig.validate` |

---

## Configuration options outside the reported run

The reported results come from `configs/default.yaml`. The settings below are
available for robustness checks and are **not** part of the published method;
`configs/extended.yaml` switches them on. Results produced with them should not
be described as reproducing the paper.

| Setting | Default | What it changes |
|---|---|---|
| `evaluation.baselines` | `[]` | adds persistence, climatology and anomaly-persistence references |
| `evaluation.peak_threshold_ugm3` | `35.0` | exceedance detection scores (POD/FAR/CSI); `null` disables |
| `evaluation.bootstrap_samples` | `0` | confidence intervals on the headline metrics |
| `features.add_day_of_year` | `false` | seasonal encoding beyond the diurnal cycle |
| `imputation.short_gap_hours` | `0` | PCHIP fill for gaps up to N hours before KNN |
| `imputation.impute_target` | `true` | `false` scores against observed labels only |
| `outliers.include_target` | `true` | `false` leaves the training target unscreened |
| `scaling.clip_scaled` | `null` | bounds scaled features against unprecedented test extremes |
| `data.min_target_completeness` | `0.0` | adds a station-admission gate |
