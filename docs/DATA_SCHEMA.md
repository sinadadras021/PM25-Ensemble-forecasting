# Input data schema

No data ships with this repository. This document specifies exactly what the
pipeline expects, so it can be pointed at any hourly air-quality archive.

## Layout

One CSV per monitoring station, all in a single directory:

```
/path/to/raw_data/
├── Aghdasieh.csv
├── District 21.csv
├── Sharif University.csv
└── ...
```

The **file stem becomes the station name**, and every station present is picked
up automatically. There is no station list to maintain: add a CSV and it joins
the study; remove one and it disappears from every table and figure.

## Columns

| column | unit | role | required |
|---|---|---|---|
| `Date` | timestamp | index | **yes** |
| `PM 2.5 ug/m3` | µg/m³ | target *and* input | **yes** |
| `O3 ppb` | ppb | input | no |
| `CO ppm` | ppm | input | no |
| `NO ppb` | ppb | input | no |
| `NO2 ppb` | ppb | input | no |
| `SO2 ppb` | ppb | input | no |
| `tmpc` | °C | meteorology | recommended |
| `relh` | % | meteorology | recommended |
| `alti` | **Pa** | meteorology | recommended |
| `vsby` | m | meteorology | recommended |
| `U` | m s⁻¹ | meteorology (wind east–west) | recommended |
| `V` | m s⁻¹ | meteorology (wind north–south) | recommended |
| `PM 10 ug/m3`, `NOx ppb`, `feel` | — | dropped by default | no |

Extra columns are used as inputs automatically. Missing optional columns are
skipped without error.

### Why three columns are dropped by default

- **`PM 10 ug/m3`** — strongly collinear with PM2.5 at these sites. Leaving it in
  lets the model read most of its own answer off a co-measured channel, which
  inflates apparent skill without improving the forecast.
- **`NOx ppb`** — equals NO + NO2 by construction. Perfect collinearity adds no
  information and destabilises any attribution.
- **`feel`** — apparent temperature, a deterministic function of temperature,
  humidity and wind, all already present.

### Units matter

In the reference archive `alti` is in **pascals** (≈ 101325) and `vsby` in
**metres**. The pipeline does not screen against fixed physical bounds — it
derives IQR bounds from your own training data — so mixed units will not crash,
but they will produce a scaler fitted to two different distributions. Keep units
consistent within a column across all stations.

### Wind

If your archive stores `wind_speed` and `wind_direction` instead of `U`/`V`, the
pipeline converts them (paper Eqs. 1–2) — set the column names under
`features.wind_speed_column` / `wind_direction_column`. When `U`/`V` are already
present the conversion is a no-op.

## Timestamps

- Hourly resolution. Missing hours may be absent rows or `NaN` rows; the pipeline
  reindexes onto a complete grid either way.
- Formats are tried in the order listed under `data.datetime_formats`, then
  inferred. Declaring the format matters: `01/02/2015` is ambiguous, and pandas'
  inference can flip between day-first and month-first *within a single file*,
  silently reordering a year of data.
- Duplicated timestamps are averaged by default; `first`, `last` and `raise` are
  available.

## Missing values and sentinels

Write missing values as empty cells or `NaN`. Sentinel strings (`---`, `NoData`)
become `NaN` during numeric coercion. Numeric sentinels such as `-999` are caught
by the IQR screen rather than by the parser.

## Example

```csv
Date,PM 2.5 ug/m3,O3 ppb,CO ppm,NO ppb,NO2 ppb,SO2 ppb,tmpc,relh,alti,vsby,U,V
2015-01-01 00:00:00,68.6,,4.9,183.9,44.4,20.1,7.0,52.7,102099.6,7998.4,0.357,-2.027
2015-01-01 01:00:00,71.2,,5.1,190.4,46.8,21.3,6.4,55.1,102105.2,7500.0,0.401,-1.884
2015-01-01 02:00:00,,,5.0,,45.1,20.8,5.9,57.3,102110.8,7200.0,0.522,-1.640
```

## Adapting to another network

1. Set `data.target` to your pollutant column.
2. Set `data.global_start` / `global_end` and both split blocks in `configs/`.
3. Review `data.drop_columns` for your archive's collinear pairs.
4. Run `pm25forecast validate-config --strict-paths`, then the `quick_test` config.

If `run-all` completes with `configs/quick_test.yaml`, the archive is readable and
the splits are sane. Only then is a full run worth starting.
