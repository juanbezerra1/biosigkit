"""Metrics for force / dynamometry traces (e.g. force-matching tasks)."""

from __future__ import annotations

import numpy as np


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
