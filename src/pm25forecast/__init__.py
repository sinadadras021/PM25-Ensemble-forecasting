"""Multi-strategy deep ensembles for 48-hour urban PM2.5 forecasting.

A configuration-driven, leakage-audited implementation of the published method:
four deep base learners per station, combined by six ensembles spanning three
strategies -- statistical averaging, non-linear meta-learning, and context-aware
selection.

Quick start
-----------
::

    pip install -e .
    pm25forecast validate-config --config configs/default.yaml --raw-dir /path/to/csvs
    pm25forecast run-all --config configs/default.yaml --raw-dir /path/to/csvs

``docs/PAPER_MAPPING.md`` maps every equation and table of the paper to the
function that implements it.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
