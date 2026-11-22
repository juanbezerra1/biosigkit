"""Core container for multichannel, uniformly sampled biosignals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, eq=False)
class Signal:
    """Multichannel signal sampled at a constant rate.

    Parameters
    ----------
    data:
        Array of shape ``(n_channels, n_samples)``. A 1-D array is treated as a single channel.
    fs:
        Sampling frequency in Hz.
    channels:
        Unique channel names. Defaults to ``ch1, ch2, ...``.
    t0:
        Time (seconds) of the first sample.
    meta:
        Free-form metadata (subject id, units, preprocessing provenance, ...).

    ``Signal`` is immutable: every processing function returns a new object, so the raw data is
    never modified by accident.
    """

    data: np.ndarray
    fs: float
    channels: tuple[str, ...] = ()
    t0: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=float)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if data.ndim != 2:
            raise ValueError(f"data must be 1-D or 2-D, got {data.ndim}-D")
        fs = float(self.fs)
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"fs must be a positive number, got {self.fs!r}")
        channels = tuple(str(c) for c in self.channels) or tuple(
            f"ch{i + 1}" for i in range(data.shape[0])
        )
        if len(channels) != data.shape[0]:
            raise ValueError(f"{len(channels)} channel names for {data.shape[0]} data rows")
        if len(set(channels)) != len(channels):
            raise ValueError("channel names must be unique")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "fs", fs)
        object.__setattr__(self, "channels", channels)

    # ------------------------------------------------------------------ basic properties
    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.n_samples / self.fs

    def times(self) -> np.ndarray:
        """Time stamp (seconds) of every sample."""
        return self.t0 + np.arange(self.n_samples) / self.fs

    # ------------------------------------------------------------------ transformations
    def with_data(
        self, data: np.ndarray, *, fs: float | None = None, t0: float | None = None
    ) -> Signal:
        """Copy with new data (and optionally a new sampling rate / start time)."""
        return replace(
            self,
            data=data,
            fs=self.fs if fs is None else fs,
            t0=self.t0 if t0 is None else t0,
            meta=dict(self.meta),
        )

    def with_meta(self, **updates: Any) -> Signal:
        """Copy with metadata entries added or replaced."""
        return replace(self, meta={**self.meta, **updates})

    def pick(self, names: Sequence[str]) -> Signal:
        """Keep only the named channels (in the requested order)."""
        idx = []
        for name in names:
            if name not in self.channels:
                raise KeyError(f"unknown channel {name!r}; available: {self.channels}")
            idx.append(self.channels.index(name))
        return replace(self, data=self.data[idx], channels=tuple(names), meta=dict(self.meta))

    def crop(self, tmin: float | None = None, tmax: float | None = None) -> Signal:
        """Keep samples with ``tmin <= t < tmax`` (absolute time in seconds)."""
        start = 0 if tmin is None else max(0, round((tmin - self.t0) * self.fs))
        stop = (
            self.n_samples
            if tmax is None
            else min(self.n_samples, round((tmax - self.t0) * self.fs))
        )
        if stop <= start:
            raise ValueError("crop window is empty")
        return self.with_data(self.data[:, start:stop], t0=self.t0 + start / self.fs)

    # ------------------------------------------------------------------ tabular conversion
    def to_frame(self, time_col: str = "time") -> pd.DataFrame:
        """Long-in-time DataFrame: one ``time`` column plus one column per channel."""
        df = pd.DataFrame(self.data.T, columns=list(self.channels))
        df.insert(0, time_col, self.times())
        return df

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        *,
        fs: float | None = None,
        time_col: str | None = None,
        channels: Sequence[str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Signal:
        """Build a signal from a DataFrame (one column per channel).

        Sampling is assumed to be uniform. If ``time_col`` is given it is only used to estimate
        (or verify) ``fs`` and to set ``t0``; non-monotonic time stamps are rejected.
        """
        if channels is None:
            channels = [
                c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])
            ]
        if len(channels) == 0:
            raise ValueError("no numeric channel columns found")
        t0 = 0.0
        if time_col is not None:
            t = df[time_col].to_numpy(dtype=float)
            if len(t) < 2 or not np.all(np.diff(t) > 0):
                raise ValueError("time column must be strictly increasing")
            est_fs = (len(t) - 1) / (t[-1] - t[0])
            if fs is None:
                fs = round(float(est_fs), 3)
            elif abs(est_fs - fs) > 0.01 * fs:
                raise ValueError(
                    f"fs={fs} Hz does not match the time column (estimated {est_fs:.3f} Hz)"
                )
            t0 = float(t[0])
        elif fs is None:
            raise ValueError("provide fs or time_col")
        data = df[list(channels)].to_numpy(dtype=float).T
        return cls(data, fs=fs, channels=tuple(channels), t0=t0, meta=dict(meta or {}))
