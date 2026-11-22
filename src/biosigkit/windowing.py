"""Segmenting signals: event tables, sliding windows and event-locked epochs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .signals import Signal

REQUIRED_EVENT_COLUMNS = ("onset", "offset", "label")


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Check and normalise an event table.

    Required columns are ``onset`` and ``offset`` (seconds) and ``label``. Events are returned
    sorted by onset and extra columns (e.g. ``level``) are kept. Events must not overlap.
    """
    missing = [c for c in REQUIRED_EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events table is missing columns: {missing}")
    out = events.copy()
    out["onset"] = out["onset"].astype(float)
    out["offset"] = out["offset"].astype(float)
    out["label"] = out["label"].astype(str)
    if not np.isfinite(out[["onset", "offset"]].to_numpy()).all():
        raise ValueError("event onsets/offsets must be finite")
    if (out["offset"] <= out["onset"]).any():
        raise ValueError("every event needs offset > onset")
    out = out.sort_values("onset", kind="stable").reset_index(drop=True)
    if (out["onset"].to_numpy()[1:] < out["offset"].to_numpy()[:-1]).any():
        raise ValueError("events must not overlap")
    return out


def sliding_windows(
    sig: Signal, win_s: float, step_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut a signal into (possibly overlapping) windows.

    Returns
    -------
    windows:
        Array ``(n_windows, n_channels, win_samples)`` (a read-only view, no copy is made).
    t_start, t_stop:
        Start / stop time (seconds, absolute) of every window.
    """
    win = round(win_s * sig.fs)
    step = round(step_s * sig.fs)
    if win < 2 or step < 1:
        raise ValueError("window must span >= 2 samples and step >= 1 sample")
    if win > sig.n_samples:
        raise ValueError(f"window ({win_s} s) is longer than the signal ({sig.duration:.3f} s)")
    view = sliding_window_view(sig.data, win, axis=1)[:, ::step, :]
    windows = np.moveaxis(view, 1, 0)
    t_start = sig.t0 + np.arange(windows.shape[0]) * step / sig.fs
    return windows, t_start, t_start + win / sig.fs


@dataclass(eq=False)
class Epochs:
    """Event-locked segments: ``data`` has shape ``(n_epochs, n_channels, n_times)``."""

    data: np.ndarray
    fs: float
    channels: tuple[str, ...]
    tmin: float
    events: pd.DataFrame

    def __len__(self) -> int:
        return int(self.data.shape[0])

    @property
    def times(self) -> np.ndarray:
        """Time axis relative to the event onset (seconds)."""
        return self.tmin + np.arange(self.data.shape[-1]) / self.fs

    def select(self, mask: np.ndarray) -> Epochs:
        """Keep epochs where ``mask`` is True."""
        keep = np.asarray(mask, dtype=bool)
        return Epochs(
            self.data[keep],
            self.fs,
            self.channels,
            self.tmin,
            self.events.loc[keep].reset_index(drop=True),
        )


def make_epochs(
    sig: Signal,
    events: pd.DataFrame,
    *,
    tmin: float,
    tmax: float,
    on_out_of_range: Literal["drop", "raise"] = "drop",
) -> Epochs:
    """Extract fixed-length epochs ``[onset + tmin, onset + tmax)`` around every event."""
    if tmax <= tmin:
        raise ValueError("tmax must be greater than tmin")
    ev = validate_events(events)
    n_times = round((tmax - tmin) * sig.fs)
    start = np.rint((ev["onset"].to_numpy() + tmin - sig.t0) * sig.fs).astype(int)
    inside = (start >= 0) & (start + n_times <= sig.n_samples)
    if not inside.all() and on_out_of_range == "raise":
        raise ValueError(f"{int((~inside).sum())} epoch(s) fall outside the recording")
    if inside.any():
        data = np.stack([sig.data[:, s : s + n_times] for s in start[inside]])
    else:
        data = np.empty((0, sig.n_channels, n_times))
    kept = ev.loc[inside].reset_index(drop=True)
    return Epochs(data, sig.fs, sig.channels, float(tmin), kept)
