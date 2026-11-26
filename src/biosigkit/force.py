"""Metrics for force / dynamometry traces (e.g. force-matching tasks)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def _inside(force: np.ndarray, low: float, high: float) -> np.ndarray:
    x = np.asarray(force, dtype=float)
    return (x >= low) & (x <= high)


def time_in_range(force: np.ndarray, fs: float, low: float, high: float) -> float:
    """Total time (s) the force stays inside ``[low, high]``."""
    return float(np.count_nonzero(_inside(force, low, high)) / fs)


def longest_run_in_range(force: np.ndarray, fs: float, low: float, high: float) -> float:
    """Longest uninterrupted stretch (s) inside ``[low, high]``."""
    inside = _inside(force, low, high)
    if not inside.any():
        return 0.0
    edges = np.diff(np.concatenate(([0], inside.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return float((stops - starts).max() / fs)


def tracking_error(force: np.ndarray, target: np.ndarray | float) -> dict[str, float]:
    """Accuracy of reproducing a target: RMSE, MAE and signed bias (same units as ``force``)."""
    err = np.asarray(force, dtype=float) - np.asarray(target, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def steadiness_cv(force: np.ndarray) -> float:
    """Coefficient of variation (std / mean) of a plateau; lower means a steadier grip."""
    x = np.asarray(force, dtype=float)
    mean = x.mean()
    return float(x.std(ddof=1) / mean) if mean != 0 and x.size > 1 else float("nan")


def plateau_window(
    force: np.ndarray,
    fs: float,
    *,
    frac: float = 0.8,
    smooth_s: float = 0.2,
    min_duration_s: float = 0.5,
) -> tuple[float, float] | None:
    """The hold phase of one force trial, as ``(start, stop)`` seconds from the first sample.

    ``force`` is the trace of a single trial (ramp up, hold, ramp down). The level of the hold is
    the median of the middle half of the trial, after smoothing with a ``smooth_s`` moving average.
    The plateau starts when the smoothed force first reaches ``frac`` of that level and ends when
    it last stays at or above it, so a brief dip during the hold does not cut it short. A lower
    ``frac`` tolerates an unsteady grip (on simulated trials with 4 to 10 % force noise, 0.9 lost
    up to a third of the hold to wobble) but lets in more of the ramps: the window then starts
    ``1 - frac`` of the ramp early and ends ``1 - frac`` of the ramp late.

    Returns ``None`` when there is no plateau: the level is not a positive number (the subject
    never squeezed), or the plateau is shorter than ``min_duration_s``. The assumption is that the
    hold fills the middle of the trial; a trial window that is mostly rest has no middle to read.
    """
    x = np.asarray(force, dtype=float)
    if x.size < 4 or not np.isfinite(x).all():
        return None
    width = max(1, round(smooth_s * fs))
    smooth = uniform_filter1d(x, size=width, mode="nearest")
    n = smooth.size
    level = float(np.median(smooth[n // 4 : n - n // 4]))
    if not np.isfinite(level) or level <= 0:
        return None
    above = np.flatnonzero(smooth >= frac * level)
    if above.size == 0:
        return None
    start, stop = above[0] / fs, (above[-1] + 1) / fs
    return (start, stop) if stop - start >= min_duration_s else None
