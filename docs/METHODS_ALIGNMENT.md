# Method alignment and explicit unresolved descriptions

This implementation follows the **current** manuscript's methodological description.
It does not reproduce or re-label results generated with a different method.
A small synthetic smoke test exercises training, prediction, analysis and SHAP.
No full experiment or reproduction of manuscript results was performed. See
`VALIDATION.md` for the scope of runtime checks.

| Manuscript item | Code | Implementation |
|---|---|---|
| §2.1/Table1 | `config.py`, `data.py` | 6 pollutants + 6 meteorological + 2 cyclical base features; no DWPC |
| §2.2, Eqs1–2 | `convert_weather` | U=−WSsin(direction), V=−WScos(direction); declared unit conversion |
| §2.2, Eqs3–4 | `engineer` | Local-hour sine/cosine; offset0–47 only in meta |
| §2.2 four-rule QC | `four_condition_screen` | Training Z calibration; all4 criteria simultaneous; counts by station/pollutant/split |
| §2.2 splits | `Config.cut`, `WindowPool` | One shared chronological cut, 70/30, 48-hour gap; full calendar2024 |
| §2.2 KNN/min–max | `prepare_network` | Training-only imputer per station; pooled training-only X/Y min–max |
| §4.1 pooled learning | `train`, `keras_sequence` | Four shared networks; windows never cross stations |
| Fig2 | `build_base` | Two recurrent sequence layers, dropout0.3, TimeDistributed Dense32ReLU/Dense1linear |
| Table2 MSE | `observed_mse` | MSE over observed labels, no IWL/EEL |
| Table3 | `callbacks`, `Config` | 100 epochs, batch512, Adam.001, ES10, reduceLR.5/5/min1e−6 |
| §4.2 historical context | `WindowPool.context` | Historical hour t0−47+j informs lead j+1; no realized future weather |
| MLP/XGB input | `flat_design` | Four separate forecasts, six weather, two cyclical, one offset |
| SA/PWE, Eqs6–8 | `fit_ensembles`, `predict_ensembles` | Exact arithmetic mean; convex SLSQP, no failed-optimizer fallback disguised as PWE |
| MLP, Eq9 | `build_meta`, `fit_ensembles` | 32/16/8 ReLU hidden layers, physical-target MSE, min–max fitted on meta train only |
| XGB, Eq10 | `fit_ensembles` | 100 trees, depth6, learning rate.1 |
| H-MOS, Eq11 | `fit_ensembles` | 24 OLS models of base-average forecast vs truth, indexed by known local target hour |
| SMS, Eqs12–13 | `fit_ensembles` | RF100/depth10; classifier chooses raw member prediction |
| Meta 80/20 | `meta_partition` | Unique issue cycles split chronologically across all stations |
| §4.4/Eqs14–18 | `analysis.py` | Daily RMSE, MAE, Pearson R, IOA, NAED; Table4 station macro average |
| SupplementS1 | `dm_test`, `holm` | Squared-loss differences by issue cycle; HAC/HLN; 8-comparison Holm family |
| SupplementS2 | `warning` | TP/FP/FN/TN, POD, FAR, FPR, F1, CSI, HSS etc.; threshold strictly >35 |
| §5.3 | `analyze` | Station metrics and two-episode station observations/predictions/errors; no invented GIS interpolation |
| §5.5/Fig9/S24 | `explain.py`, `plots.py` | Actual pooled Bi-GRU and MLP SHAP; separate bar/beeswarm, matched Arial fonts |
| §5.6/Fig10 | `ablations` | No weather, no cyclical, no offset; Full baseline is the exact Table4 artifact |

## Descriptions requiring author clarification

1. **SMS inputs conflict within the manuscript.** Table2 includes base forecasts,
   meteorology, cyclical time and offset; the text of the SMS paragraph describes
   only meteorology and cyclical time. Code follows Table2. Reconcile the text.
2. **ReLU placement.** Figure2 places it in TimeDistributed Dense32. Recurrent
   cells retain their standard tanh activations. Do not describe them as ReLU
   RNN cells without explicitly changing the figure and method.
3. **Unspecified preprocessing details.** KNN k5/uniform, population-std Z-scores,
   minimum2 stations for a city change, per-station imputers, observed-only
   verification labels and18/24 daily coverage are documented choices, not
   recovered historical settings. The centered detector delays its latest flag
   by one hour to avoid looking past forecast issue time.
4. **Historical context alignment.** The prose does not distinguish last-value
   persistence, a48-hour average or a lag-aligned historical trajectory. Code
   uses the latter and records observation issue times. State that rule.
5. **Prediction domain.** The output layer is linear as drawn. Reported physical
   forecasts are clipped at zero; there is no arbitrary500µg/m³ upper cap.
6. **Meta/statistical fitting.** MLP/XGB/SMS use the first80% of held-out base
   validation cycles, with the last20% held out. PWE and H-MOS use that whole
   base-validation period, following their validation-calibration equations.
7. **Grid search provenance is absent.** The supplied fixed settings are not
   evidence of a completed search. The prospective grid utility cannot recover
   original trial logs or unspecified per-ensemble optimization.
8. **Time/units remain source contracts.** Ambiguous DST labels, synoptic reporting
   latency and the old merged meteorology clock cannot be established from model
   source code. The default input-alignment guard deliberately blocks an
   unverified merged dataset.
9. **Old numbers do not validate the new method.** Earlier runs used station-specific
   models, custom losses, a different preprocessing/stacking/context pipeline and
   an average-of-four meta input. Their metrics, significance tests and SHAP
   explanations must not be presented as outputs of this candidate.

The manuscript and Supplementary Word files are external reference documents;
the pipeline never edits them.
