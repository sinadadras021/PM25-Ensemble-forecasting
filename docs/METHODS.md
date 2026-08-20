# Methods

What the pipeline does, **why each choice was made**, and — in the final section —
what it does *not* do. The limitations section is the most important part: a
method paper that lists none is either trivial or dishonest.

---

## 1. Problem statement

Given 48 hours of hourly observations at an urban monitoring station, forecast
the next **48 hourly PM2.5 concentrations**, then aggregate to daily means and
report **Day 1** (lead 1–24 h) and **Day 2** (lead 25–48 h) separately.

Everything on the input side is observed at or before issue time. No future
meteorology, no reanalysis. That makes the reported skill directly comparable to
persistence, and *not* comparable to models that consume NWP output — those enjoy
a perfect-prognosis advantage that is often left unstated.

---

## 2. Preprocessing

### 2.1 Split before fitting anything

The order of stage 1 is the entire leakage argument:

```
read → hourly grid → SPLIT → fit IQR screen → fit imputer → features → fit scaler
```

Splitting first is what makes "fitted on training data only" a *structural*
property rather than a promise in a comment. Every fitted object
(`IQRScreen`, `TrainFittedKNNImputer`, `SplitScaler`) takes the training block in
`fit` and can only `transform` anything else, and the config **rejects**
`fit_on: development` for all three.

### 2.2 Hourly grid regularisation

Missing hours become all-NaN rows before anything counts hours. Without this,
`interpolate(limit=3)` counts three *rows*, which on a sparse index can span
three days — a "3-hour" policy that quietly becomes a 3-day one.

### 2.3 IQR screening voids, it does not clip

Bounds are `Q1 − 1.5·IQR` and `Q3 + 1.5·IQR`, fitted on training data. Values
outside them become `NaN` and are then imputed.

Clipping to the boundary is the tempting shortcut and it is wrong twice over. It
converts a detectable instrument fault into a plausible-looking observation; and
applied to the **target** it truncates exactly the pollution episodes the model
exists to predict, so the metrics flatter the model precisely on the days that
matter most.

**Validation and test labels are never screened**, whatever the configuration
says. A ground truth that has been altered is no longer a ground truth.

### 2.4 KNN imputation

k = 5, multivariate, fitted on the training block. Pollutants and meteorology
co-vary, so a missing NO2 hour is better reconstructed from the concurrent NO, CO
and temperature than from a linear ramp.

The label and the encoder's view of the target are separated:

- `<target>_input` is always gap-filled — a window must be complete to exist, so
  one missing input hour would otherwise cost the 96-hour window around it;
- `<target>` is the label, reconstructed or observations-only depending on
  `imputation.impute_target`.

Keeping them apart is what makes the strict setting usable at all: with a single
column, refusing to reconstruct the label also destroys the inputs.

### 2.5 Features

| feature | why |
|---|---|
| gap-filled PM2.5 history | the strongest single predictor, fully available at issue time |
| O3, NO, NO2, CO, SO2 | co-emitted tracers and secondary-aerosol precursors |
| temperature, RH, pressure, visibility | boundary-layer and hygroscopic-growth proxies |
| U, V wind components | ventilation and transport, continuous everywhere |
| `Hour_sin`, `Hour_cos` | hour 23 and hour 00 are one hour apart in the world and 23 units apart on a number line |
| `Hour_offset` (0–47) | lets a meta-learner model lead-time decay explicitly |

### 2.6 Segmentation

A gap longer than `windows.max_gap_hours` ends the current segment, and windows
are built strictly inside one segment. Two things can break contiguity and both
are checked: a `NaN` in a needed column, and **a hole in the timestamp index**.
The second is the subtle one — once unusable rows have been deleted a frame
contains no `NaN` at all, so a NaN-only rule reports one long clean segment while
the timestamps jump across the deleted hours.

---

## 3. Splits

```
|<---------------- development 2015-2023 ---------------->|<-- test 2024 -->|
|<--- 70% training --->| 48h gap |<--- 30% validation --->|
```

**Why validation is 30 %.** It is not only an early-stopping monitor — the base
learners' predictions on it are the *entire* training set for all six ensembles.
A thin hold-out would leave the meta-learners fitting noise.

**Why the gap is 48 hours.** With a 48-hour input window and a 48-hour horizon, a
window anchored at the last training hour occupies 95 further hours. Without a
purge, the final training windows and the first validation windows overlap almost
completely, and early stopping would be reading data the model had just been
fitted to. The config warns if the gap is set below the horizon.

**Test stride.** `windows.stride.test` must be ≥ the horizon, or the config is
rejected: overlapping test windows count the same observation many times and the
"sample size" becomes fiction.

---

## 4. Base learners

Two families, differing in **both** structure and objective (paper Table 2):

| family | structure | loss |
|---|---|---|
| LSTM, GRU | 64 → 32, ReLU | inverse-weighted (IWL, Eq. 19) |
| Bi-LSTM, Bi-GRU | 2 × 32, ReLU | exponential extreme (EEL, Eq. 20) |

**Why two objectives.** A symmetric squared error trained on a right-skewed
pollutant series learns to predict the middle of the distribution: it scores well
on the many ordinary hours and systematically under-shoots the few episodes that
matter. IWL weights *low* concentrations more, protecting the clean end from a
model that drifts high everywhere. EEL weights *high* concentrations
exponentially, making one member deliberately recall-oriented. Pairing the deeper
narrowing stack with IWL and the shallower bidirectional stack with EEL produces
genuinely decorrelated errors — the precondition for any ensemble to add value.

**Scale trap, guarded in code.** Both losses operate in scaled target space where
`y ∈ [0, 1]`, bounding the EEL weight at `e¹ ≈ 2.72`. Applying the same formula to
raw µg/m³ gives `e³⁰⁰`, which overflows to `inf` and destroys the run in one step;
`resolve_loss` raises rather than allowing it.

**Direct multi-output head.** A single dense layer emits all 48 steps, rather than
an autoregressive decoder unrolled 48 times which would accumulate its own error.

---

## 5. Ensembles

Six models, three strategies — see `docs/PAPER_MAPPING.md` §4.2 for the
equation-by-equation mapping and `models/ensembles.py` for the reasoning behind
each design choice. Two properties are worth restating here:

**All six are fitted on the validation block.** Fitting a stack on in-sample base
predictions is the classic failure of stacked generalisation: the members look
near-perfect on data they memorised, the meta-learner learns to trust whichever
one overfit hardest, and the structure collapses on anything new. Because the
mistake is invisible in the training metrics, a guard re-derives the provenance of
the arrays being used.

**SA is the control.** It has no parameters and cannot overfit. An elaborate
meta-learner that ties with SA has demonstrated nothing, and the pipeline reports
SA alongside the rest so that comparison is unavoidable.

---

## 6. Evaluation

**Daily means, Day 1 vs Day 2.** A single 48-hour average hides the decay of
predictability with lead time, and it is precisely the Day-2 number that decides
whether a system is useful for planning rather than only for reacting.

**Five metrics** (Eqs. 14–18): RMSE, MAE, R (Pearson), IOA (Willmott), Unc%.

R and IOA are reported together on purpose. A forecast that tracks every rise and
fall but sits 20 µg/m³ too high has R ≈ 1 and a poor IOA; seeing both distinguishes
"wrong shape" from "right shape, wrong level". Unc% measures the *spread* of the
error about the MAE, so a perfectly consistent bias scores zero uncertainty
despite a large MAE — which is exactly the diagnostic you want when deciding
whether a model needs recalibration or retraining.

**A daily mean requires ≥ 18 observed hours**, otherwise it is `NaN` rather than a
silently short average.

---

## 7. Assumptions and limitations

### 7.1 No numerical weather prediction

Inputs stop at issue time. The reported skill is therefore a *lower* bound for an
operational system with access to NWP output, and it is not directly comparable
with published models that consume future meteorology.

### 7.2 Station-specific models

Each station gets its own four base learners and six ensembles. Tehran's sites
differ in traffic exposure, altitude and local emission mix, and a pooled model
would learn the city mean and miss the station-level episodes a warning system
exists for. The costs are more artefacts to manage and no transfer to a station
with too little history. A pooled or fine-tuned variant is a reasonable extension
and is not implemented here.

### 7.3 Min-max scaling extrapolates on unprecedented extremes

A test value above the training maximum maps above 1 and the network must
extrapolate. This is *honest* — it is what an operational model faces — but it
degrades peak prediction in a record-breaking year. Mitigations available:
`scaling.method: robust`, and `scaling.clip_scaled`.

### 7.4 The ensembles see one year of meta-training data per station

Six ensembles fitted on the same validation block will, collectively, have
explored it thoroughly by the time the best is selected. The test year is touched
once and never used for selection, so the reported numbers are honest — but the
*choice of best strategy* carries more uncertainty than any single row's
confidence interval suggests. Treat the strategy ranking as indicative unless it
holds across most stations.

### 7.5 Determinism is per-machine

`deterministic: true` pins seeds and enables deterministic kernels, but cuDNN
chooses different reduction orders on different GPUs. Bit-identical results across
machines are not guaranteed. `seeding.environment_fingerprint()` records the
Python, NumPy, TensorFlow and GPU details into `base_model_meta.json`.

### 7.6 What is not modelled

No spatial term (each station is independent — no graph, no kriging), no chemical
transport model coupling, no explicit treatment of dust intrusion or regional
transport, and no quantification of *observation* uncertainty. Unc% describes the
consistency of the forecast error, not instrumental error.

---

## References

- Willmott, C. J. (1981). On the validation of models. *Physical Geography*, 2(2), 184–194.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Glahn, H. R., & Lowry, D. A. (1972). The use of Model Output Statistics (MOS) in objective weather forecasting. *Journal of Applied Meteorology*, 11(8), 1203–1211.
- Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192–213.
