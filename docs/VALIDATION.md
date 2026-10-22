# Validation record — 22 October 2026

The final local synthetic suite passed **23 tests in 38.12 seconds** on
Python 3.10.11 / TensorFlow 2.10.1. This is an execution check, not a reproduction
of the manuscript's reported accuracy or a guarantee that no defect remains.

## Executed checks

- The four actual Figure 2 architectures each fit for one epoch on a small
  artificial two-station dataset. All six ensembles are fitted and evaluated.
  Research layer sizes remain intact; only test data volume, epochs and tree
  counts are reduced. The paper configuration remains unchanged.
- Saved H5 models reload and reproduce predictions on repeated inference.
  Resume preserves all five neural checkpoint hashes. Config changes and
  deliberately corrupted checkpoints are rejected.
- SA equals the member mean; PWE weights are convex and reproduce its forecast;
  SMS selects an actual member forecast. Daily station metrics, complete-panel
  city warnings and all eight DM/Holm comparison rows are produced.
- Missing test labels survive preparation. Changing future weather/observations
  does not alter the current window or its meta context; future missing labels
  do not remove forecast cycles. The issue-hour row uses observations without an
  unavailable centered-QC decision. Meta training precedes meta validation.
- Four simultaneous QC rules detect an isolated artificial station spike while
  retaining a synchronized city episode. Test changes do not affect Z calibration.
  Timezone conversion, duplicate timestamps, unit conversion and delayed
  meteorology availability are checked independently.
- Known numerical examples verify warning counts, the strict >35 threshold,
  continuous scores, Holm adjustment and degenerate DM cases.
- Real Gradient SHAP for Bi-GRU Day 1/2 and real Kernel SHAP for MLP run with a
  very small test budget. Shapes, finite attributions and MLP additivity are
  checked. All three separate plot types export PNG/JPG/PDF at test-only low DPI.
  Research plotting still uses 600 DPI.

## Defect repaired

The original candidate defined and registered its Keras loss class afresh for
each network. TensorFlow 2.10 stopped when the second network attempted to
register the same class name. The loss class is now registered once through a
cached class factory; each model receives a fresh loss instance. The sequential
four-network test covers this failure. Its observed-label MSE formula is unchanged.

## Limits

The tests do not fit the full 2015–2023 dataset, evaluate scientific accuracy on
2024, establish SHAP convergence, run the complete ablation/grid search, assess
multi-seed variability or validate a real station's source clocks/device ranges.
Tiny Kernel SHAP backgrounds can be rank-deficient; that test checks execution
and additivity, not a publishable feature ranking. Local dependency deprecation
warnings remain visible rather than being globally suppressed.

Before a scientific run, resolve the input clock/latency contracts and the
manuscript ambiguities listed in `METHODS_ALIGNMENT.md` (notably SMS input prose,
context alignment and missing historical grid provenance). Numerical results
from the older implementation must not be attributed to this revised method.
CI is configured to repeat the same synthetic suite on CPU TensorFlow 2.15.1;
the Actions result is a separate check from this local record.
