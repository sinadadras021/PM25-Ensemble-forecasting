"""Lazy station-contained windows; pooling never joins different stations in time."""

from __future__ import annotations
import numpy as np
import pandas as pd


class WindowPool:
    """Lazy 48-hour windows sorted by forecast cycle, then by station."""

    def __init__(self, prepared, cfg, split):
        self.prepared = prepared
        self.cfg = cfg
        self.split = split
        self.names = sorted(prepared["stations"])
        records = []
        for station_id, station in enumerate(self.names):
            item = prepared["stations"][station]
            idx = item["index"]
            n = len(idx)
            if split == "train":
                lower = cfg.timestamp(cfg.development_start)
                upper = cfg.train_end
                stride = cfg.train_stride
            elif split == "val":
                lower = cfg.cut
                upper = cfg.timestamp(cfg.development_end)
                stride = cfg.validation_stride
            elif split == "test":
                lower = cfg.timestamp(cfg.test_start)
                upper = cfg.timestamp(cfg.test_end)
                stride = cfg.test_stride
            else:
                raise ValueError("Unknown split.")
            first = int(idx.searchsorted(lower))
            # Val/train histories remain inside their split. Test is allowed to
            # use the last 48 observed hours of 2023, without using a 2024 target.
            t0 = first if split == "test" else first + cfg.input_hours
            if split != "train":
                while t0 < n and idx[t0].tz_convert(cfg.pollutant_timezone).hour != 0:
                    t0 += 1
            for target in range(t0, n - cfg.horizon_hours + 1, stride):
                history = target - cfg.input_hours
                last = target + cfg.horizon_hours - 1
                if history < 0 or idx[last] > upper:
                    break
                if idx[target] < lower:
                    continue
                if idx[last] - idx[history] != pd.Timedelta(hours=95):
                    raise ValueError("Window crosses an hourly discontinuity.")
                # Prediction admission must never depend on unavailable future
                # observations. Missing test truth is handled after prediction.
                if split != "test":
                    counts = (
                        np.isfinite(item["y"][target : target + 48])
                        .reshape(2, 24)
                        .sum(axis=1)
                    )
                    if np.any(counts < cfg.min_target_hours_per_day):
                        continue
                records.append((idx[target].value, station_id, history, target))
        records.sort(key=lambda r: (r[0], r[1]))
        self.records = np.asarray(records, dtype="int64").reshape(-1, 4)
        if len(self) == 0:
            raise ValueError(
                f"No admissible {split} windows; inspect coverage and time contract."
            )
        absent = set(range(len(self.names))) - set(self.records[:, 1])
        if absent:
            raise ValueError(
                f"{split}: no admissible windows for {[self.names[i] for i in sorted(absent)]}; do not silently report all stations."
            )

    def __len__(self):
        return len(self.records)

    @property
    def first_targets(self):
        return pd.to_datetime(self.records[:, 0], utc=True)

    def sample(self, i):
        _, s, start, target = self.records[i]
        item = self.prepared["stations"][self.names[int(s)]]
        x = item["x"][start:target].copy()
        x[-1] = item["last_x"][target - 1]
        # All channels at the issue hour avoid the unavailable centered-QC flag.
        y = item["scaled_y"][target : target + 48, None].copy()
        return x, y

    def batches(self, batch_size):
        for lo in range(0, len(self), batch_size):
            samples = [
                self.sample(i) for i in range(lo, min(lo + batch_size, len(self)))
            ]
            yield np.stack([p[0] for p in samples]), np.stack([p[1] for p in samples])

    def physical_y(self):
        return np.stack(
            [
                self.prepared["stations"][self.names[int(s)]]["y"][t : t + 48]
                for _, s, _, t in self.records
            ]
        )

    def context(self):
        wanted = [c for c in self.cfg.context if c != "hour_offset"]
        positions = [self.cfg.features.index(c) for c in wanted]
        contexts = []
        for _, s, start, target in self.records:
            item = self.prepared["stations"][self.names[int(s)]]
            history = item["physical"][start:target].copy()
            history[-1] = item["last_physical"][target - 1]
            # Lead j receives historical hour t0-47+j, never weather at t0+1+j.
            block = history[:, positions]
            if "hour_offset" in self.cfg.context:
                block = np.column_stack([block, np.arange(48)])
            contexts.append(block)
        return np.stack(contexts).astype("float64")

    def local_target_hours(self):
        ns = (
            self.records[:, 0, None]
            + np.arange(48, dtype="int64")[None, :] * 3600 * 10**9
        )
        return (
            pd.DatetimeIndex(pd.to_datetime(ns.ravel(), utc=True))
            .tz_convert(self.cfg.pollutant_timezone)
            .hour.to_numpy()
            .reshape(-1, 48)
        )

    def exact_indices(self):
        out = []
        for time, s, start, target in self.records:
            idx = self.prepared["stations"][self.names[int(s)]]["index"]
            out.append(
                dict(
                    station=self.names[int(s)],
                    split=self.split,
                    history_start=idx[start],
                    issue_time=idx[target - 1],
                    first_target=idx[target],
                    last_target=idx[target + 47],
                    history_end=idx[target - 1],
                    context_last_observation=idx[target - 1],
                )
            )
        return pd.DataFrame(out)


def keras_sequence(pool, batch_size):
    import tensorflow as tf

    class Sequence(tf.keras.utils.Sequence):
        def __len__(self):
            return (len(pool) + batch_size - 1) // batch_size

        def __getitem__(self, batch):
            samples = [
                pool.sample(i)
                for i in range(
                    batch * batch_size, min((batch + 1) * batch_size, len(pool))
                )
            ]
            return np.stack([x for x, _ in samples]), np.stack([y for _, y in samples])

    return Sequence()


def meta_partition(pool, ratio):
    """Hold out complete later cycles, including every station in each cycle."""
    anchors = np.unique(pool.records[:, 0])
    cut = int(len(anchors) * ratio)
    if cut <= 0 or cut >= len(anchors):
        raise ValueError(
            "Insufficient distinct validation cycles for chronological meta split."
        )
    threshold = anchors[cut]
    train = pool.records[:, 0] < threshold
    valid = ~train
    if not train.any() or not valid.any():
        raise ValueError("Empty meta train/validation partition.")
    return train, valid
