# PM2.5 Ensemble Forecasting

[![Runtime checks](https://github.com/sinadadras021/PM25-Ensemble-forecasting/actions/workflows/static-check.yml/badge.svg)](https://github.com/sinadadras021/PM25-Ensemble-forecasting/actions/workflows/static-check.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Four pooled recurrent networks and six ensemble strategies for 48-hour urban
PM2.5 forecasts. The implementation follows the methodological description in
`PM2.5 Paper_Hajighasemi and Dadras.docx` reviewed on 22 October 2026.

**Validation scope:** the source has been exercised with a small synthetic
runtime test, including actual one-epoch neural fits, model reloads, prediction,
analysis and both SHAP explainers. This checks execution and selected invariants;
it does not reproduce the paper's results or prove the absence of every bug.
No full research training was performed for this release. Earlier implementations
remain in Git history; their saved metrics are not outputs of this version.

## Start here

| Task | Reference |
| --- | --- |
| Understand the source layout | [Project guide](PROJECT_GUIDE.txt) |
| Trace manuscript methods to code | [Method alignment](docs/METHODS_ALIGNMENT.md) |
| Inspect test scope and limitations | [Validation record](docs/VALIDATION.md) |
| Configure a research run | [Paper configuration](configs/paper.json) |

**Source-only repository:** datasets, trained weights, generated predictions and
manuscript files are not distributed here. Automated tests generate artificial
data in a temporary directory.

## Implemented method

- Four pooled city-wide networks, not four networks per station. Windows are
  constructed within each station before pooling; no sequence crosses stations.
- Figure 2: two recurrent layers, 0.3 dropout after each, TimeDistributed Dense32
  with ReLU, then TimeDistributed Dense1 with a linear output. Bi-RNNs have 32
  units per direction in each layer; ordinary RNNs use 64 then 32 units. The
  Table 2 ReLU description applies to the explicitly drawn Dense head, not an
  undocumented change to recurrent-cell activations.
- Table 2: MSE for all four base networks. Missing labels contribute no squared
  error; they are not replaced by synthetic ground truth.
- Table 3: 48-hour history, 48-hour horizon, Adam 0.001, 100 epochs maximum,
  batch512, early-stopping patience10, reduce-LR factor0.5/patience5/minimum1e-6,
  seed42. Early stopping restores the best validation weights.
- Six pollutants, six meteorological features, Hour_sin and Hour_cos: 14 base
  inputs. DWPC, PM10, NOx and `feel` are not model inputs.
- Four simultaneous Z-score rules, training-only mean/std, applied to each
  pollutant and station with a simultaneous network increase. No IQR capping.
- 70/30 chronological development split with a 48-hour gap; 2024 is independent.
  Base-held-out validation predictions alone supply the meta dataset. Meta
  train/validation is 80/20 by **unique forecast cycle**, shared across stations.
- Four separate base forecasts plus historical meteorology, historical cyclical
  time and Hour_offset give 13 meta features. No realized weather from a
  forecast's future window is supplied to the meta model.
- SA, SLSQP-constrained PWE, MLP32/16/8, XGB100/depth6/lr0.1, 24 H-MOS
  regressions, and RF100/depth10 SMS. The SMS feature choice follows Table 2;
  paragraph P091 currently gives a narrower description and needs clarification.
- Daily station metrics and their arithmetic station averages form Table4.
  City warnings use complete station panels. DM uses daily squared-error
  differences averaged across stations by issue cycle, HAC, HLN and Holm across
  eight comparisons. Coverage is measured, never assumed to be181 cycles.
- Test forecast admission never inspects future target availability. Missing
  verification labels are handled only during subsequent daily scoring.
- SHAP explains the actual Bi-GRU and actual MLP, including all four individual
  base inputs. Feature ablations prepare and fit their own feature-specific
  imputer/scaler; Full is read from the same final prediction artifact.

## Data contract — resolve before a future run

The source archive's pollutant labels are local Tehran time. The historical DST
interpretation and the old merged meteorology clock are not established by a
CSV header. Consequently `configs/paper.json` deliberately has
`merged_meteorology_alignment_verified: false`. This is an **input-provenance
gate**, not a claim that the old merged file is already aligned. Provide a
separate meteorology CSV with its actual timezone, or verify the merged clock
before enabling that flag. Unknown ambiguous/nonexistent local times raise a
clear error instead of being silently shifted or guessed.

One station CSV per file must contain `Date`, `PM 2.5 ug/m3`, `O3 ppb`, `CO ppm`,
`NO ppb`, `NO2 ppb` and `SO2 ppb`. Weather columns are `tmpc`, `relh`, `alti`,
`vsby`, `U`, `V`, or `wind_speed`/`wind_direction` in place of U/V. If a separate
meteorology file is used it follows the same declared timestamp-column/format
contract; rename an IEM `valid` column explicitly before use. Declare units:
visibility m/km/mile; pressure hPa/inHg; wind m/s/knots. Internal units are km,
hPa and m/s. Pollutant units must already match their column names. Backward
ASOF matching never selects a weather report with a later timestamp.

No instrument measurement ranges or pollutant-specific maximum ceilings are
invented. Negative concentrations and nonfinite entries become missing values.
Z-score calibration is station/pollutant specific, uses population std, and a
city increase requires at least two observed station increases. Undefined ratios
or incomplete centered triples do not satisfy all four rules.

**Centered QC has a one-hour availability delay.** At issue time t0 a centered
decision for t0 is unavailable. Each input window restores its final row from
the unscreened observations, using only its training-fitted imputer. Flags for
earlier historical rows are already available. This prevents an otherwise
hidden dependence on the first future observation. Validation/test verification
labels are preserved; forecast labels are never imputed. The default18/24 daily
observation requirement, KNN k5/uniform, masked-label handling, exact historical
context alignment and nonnegative forecast clipping are explicit implementation
choices where the prose is incomplete. They must be described if used.

Observation timestamp is not guaranteed to equal reporting availability time.
Reporting latency, device ranges, historical DST labels and raw merged-weather
provenance remain data-owner questions. No operational guarantee is made.
For a separate weather file, `meteorology_reporting_delay_minutes` shifts the
matching index to report-availability time; its default0 is an explicit unverified
latency assumption. A pre-merged file must already respect availability at issue.

## Install and future commands

```bash
git clone https://github.com/sinadadras021/PM25-Ensemble-forecasting.git
cd PM25-Ensemble-forecasting
python -m pip install -e .
# Edit configs/paper.json: raw directory, NEW output directory, source time/units.
pm25forecast validate-config --config configs/paper.json
pm25forecast prepare --config configs/paper.json
pm25forecast train --config configs/paper.json
pm25forecast predict --config configs/paper.json
pm25forecast analyze --config configs/paper.json
pm25forecast explain --config configs/paper.json
pm25forecast plot-shap --config configs/paper.json
pm25forecast ablation --config configs/paper.json
```

The automated tests use artificial data only. They reduce epochs, training
sample count, tree count, SHAP budget and raster DPI without changing the
research defaults. Run them with:

```bash
pip install -e '.[test]'
python -m pytest -q --durations=8
```

See [the validation record](docs/VALIDATION.md) for tested scope and limitations.
Windows native GPU support depends on the TensorFlow version and compatible
CUDA/cuDNN installation. `requirements-research.txt` records the environment used
for local tests; installing that file alone does not install this package.
The package supports Python3.10/3.11 and TensorFlow2.10–2.15. CI checks CPU runtime
with TensorFlow2.15.1; local tests use the research TensorFlow2.10.1 environment.

A future train command fits four pooled base models and six pooled ensembles.
It does not generate test forecasts until `predict` is explicitly requested.
Prepared/model manifests contain configuration, code and input hashes;
incompatible cached artifacts are refused. Input/Word files are never written.

## Grid search and manuscript provenance

The manuscript claims a completed historical grid search but neither its grid
nor trial log was supplied. `grid` is empty for that reason. `grid-search`
requires at least two declared learning-rate/dropout combinations, logs their
validation-only scores and emits a selected configuration. It is a prospective
search utility, not evidence that the historical search happened, and does not
reconstruct separate original ensemble tuning. Do not describe a one-point
fixed configuration or invented grid as a reproduced optimization experiment.

The earlier implementation and the current manuscript describe different
methods. See [the alignment notes](docs/METHODS_ALIGNMENT.md) before interpreting
any manuscript result. Numerical results obtained with the older method do not
validate this implementation.

## Citation and license

See [CITATION.cff](CITATION.cff) for software citation and [LICENSE](LICENSE)
for the MIT license.

Author: [Sina Dadras](https://github.com/sinadadras021) · [CV](https://github.com/sinadadras021/CV)
