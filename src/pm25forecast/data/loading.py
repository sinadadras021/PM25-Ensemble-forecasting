"""Station discovery and robust CSV ingestion.

The raw archive is a directory of one CSV per monitoring station. Nothing about
station names, ordering or count is hard-coded: add a station file and it is
picked up on the next run; remove one and it disappears from every table and
figure. That property is what makes this repository a *template* rather than a
transcript of one particular experiment.
"""

from __future__ import annotations

import glob
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..validation import IntegrityError, check_columns_present, check_no_duplicate_timestamps

logger = get_logger(__name__)

__all__ = ["StationFile", "slugify", "discover_stations", "read_station_csv", "coerce_datetime_index"]


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StationFile:
    """A discovered station and where its raw data lives."""

    name: str  #: human-readable name, taken from the file stem
    slug: str  #: filesystem/URL-safe identifier derived from ``name``
    path: Path

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name} ({self.path.name})"


def slugify(text: str, *, max_length: int = 80) -> str:
    """ASCII-safe identifier that survives Windows paths, URLs and HDF5 keys.

    Non-Latin station names (Persian, in the reference dataset) are transliterated
    where possible and otherwise replaced by a stable hash suffix, so two distinct
    stations never collapse onto the same slug.

    >>> slugify("Sharif University")
    'Sharif_University'
    >>> slugify("District 21")
    'District_21'
    """
    normalised = unicodedata.normalize("NFKD", text)
    ascii_text = normalised.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_")
    if not slug:
        # Entirely non-Latin name: keep it addressable and unique.
        digest = abs(hash(text)) % (10**8)
        slug = f"station_{digest:08d}"
    return slug[:max_length]


def discover_stations(
    raw_dir: str | Path, pattern: str = "*.csv", *, exclude: Sequence[str] = ()
) -> List[StationFile]:
    """List every station file under ``raw_dir``, sorted for deterministic runs.

    Parameters
    ----------
    exclude:
        Case-insensitive substrings; any matching filename is skipped. Useful for
        keeping notes or template files beside the data.

    Raises
    ------
    FileNotFoundError
        If the directory is missing or contains no matching file -- an empty run
        that reports "0 stations processed" at the very end wastes more time than
        an immediate error.
    """
    root = Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {root}")

    matches = sorted(Path(p) for p in glob.glob(str(root / pattern)))
    lowered = tuple(e.lower() for e in exclude)
    matches = [
        p
        for p in matches
        if p.is_file() and not any(token in p.name.lower() for token in lowered)
    ]

    if not matches:
        raise FileNotFoundError(
            f"No files matching {pattern!r} in {root}. "
            f"Check paths.raw_dir and data.station_glob."
        )

    stations: List[StationFile] = []
    seen_slugs: Dict[str, str] = {}
    for path in matches:
        name = path.stem
        slug = slugify(name)
        if slug in seen_slugs:
            raise IntegrityError(
                f"Station slug collision: {name!r} and {seen_slugs[slug]!r} both map to "
                f"{slug!r}. Rename one of the files."
            )
        seen_slugs[slug] = name
        stations.append(StationFile(name=name, slug=slug, path=path))

    logger.info("Discovered %d station files in %s", len(stations), root)
    return stations


# --------------------------------------------------------------------------- #
def coerce_datetime_index(
    series: pd.Series, formats: Sequence[str], *, context: str = ""
) -> pd.DatetimeIndex:
    """Parse a timestamp column, trying explicit formats before inference.

    Why not just call ``pd.to_datetime`` with ``infer_datetime_format``?
    Because ``01/02/2015`` is ambiguous and pandas' inference can flip between
    day-first and month-first *within a single file*, silently reordering a year
    of data. Trying declared formats first, and requiring that a format parse at
    least 99% of rows, removes that failure mode.
    """
    raw = series.astype("string")
    best_index: Optional[pd.DatetimeIndex] = None
    best_rate = 0.0
    best_format: Optional[str] = None

    for fmt in list(formats) + [None]:
        try:
            parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
        except (ValueError, TypeError):
            continue
        rate = float(parsed.notna().mean())
        if rate > best_rate:
            best_rate, best_index, best_format = rate, pd.DatetimeIndex(parsed), fmt
        if rate >= 0.999:
            break

    if best_index is None or best_rate == 0.0:
        raise IntegrityError(
            f"{context}: could not parse any timestamp in column with formats {list(formats)}. "
            f"First values: {raw.head(3).tolist()}"
        )
    if best_rate < 0.99:
        logger.warning(
            "%s: best timestamp format %r parsed only %.2f%% of rows; %d rows will be dropped.",
            context,
            best_format,
            best_rate * 100,
            int((~best_index.notna()).sum()),
        )
    else:
        logger.debug("%s: parsed timestamps with format %r", context, best_format)
    return best_index


def read_station_csv(
    station: StationFile,
    *,
    datetime_column: str = "Date",
    datetime_formats: Sequence[str] = ("%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S"),
    duplicate_policy: str = "mean",
) -> pd.DataFrame:
    """Read one station into a numeric, time-indexed, de-duplicated frame.

    Steps, in order:

    1. read as text (no premature date parsing that could mis-order rows);
    2. parse the timestamp column with :func:`coerce_datetime_index`;
    3. drop unparseable timestamps;
    4. coerce every remaining column to numeric (``errors='coerce'``, so sentinel
       strings such as ``"---"`` or ``"NoData"`` become NaN rather than crashing);
    5. resolve duplicated timestamps;
    6. sort.

    Parameters
    ----------
    duplicate_policy:
        ``"mean"`` averages colliding rows (right for two sensors reporting the
        same hour), ``"first"``/``"last"`` keep one, ``"raise"`` refuses to guess.
    """
    frame = pd.read_csv(station.path, dtype=str, keep_default_na=True)

    if datetime_column not in frame.columns:
        candidates = [c for c in frame.columns if re.search(r"date|time", c, re.I)]
        if not candidates:
            raise IntegrityError(
                f"{station.name}: no column named {datetime_column!r} and no date-like "
                f"alternative among {list(frame.columns)}."
            )
        logger.warning(
            "%s: column %r not found; using %r instead.",
            station.name, datetime_column, candidates[0],
        )
        datetime_column = candidates[0]

    index = coerce_datetime_index(
        frame[datetime_column], datetime_formats, context=station.name
    )
    frame = frame.drop(columns=[datetime_column])
    frame.index = index

    unparsed = int(frame.index.isna().sum())
    if unparsed:
        logger.warning("%s: dropping %d rows with unparseable timestamps.", station.name, unparsed)
        frame = frame[frame.index.notna()]

    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_index()

    frame = _resolve_duplicates(frame, policy=duplicate_policy, context=station.name)
    check_no_duplicate_timestamps(frame, context=f"{station.name} after de-duplication")

    logger.info(
        "%s: loaded %d rows x %d columns, %s -> %s",
        station.name, len(frame), frame.shape[1],
        frame.index.min() if len(frame) else "-",
        frame.index.max() if len(frame) else "-",
    )
    return frame


def _resolve_duplicates(df: pd.DataFrame, *, policy: str, context: str) -> pd.DataFrame:
    duplicated = df.index.duplicated(keep=False)
    n_dup = int(duplicated.sum())
    if n_dup == 0:
        return df

    logger.warning(
        "%s: %d rows share a timestamp with another row; policy=%r", context, n_dup, policy
    )
    if policy == "raise":
        raise IntegrityError(
            f"{context}: {n_dup} duplicated timestamps and duplicate_policy='raise'."
        )
    if policy == "mean":
        return df.groupby(level=0).mean()
    if policy in {"first", "last"}:
        return df[~df.index.duplicated(keep=policy)]
    raise ValueError(f"Unknown duplicate_policy: {policy!r}")
