"""Signal-quality report: non-finite samples, flat lines, clipping and mains contamination.

Run it on the **raw** recording, before any filtering: clipping shows up as many samples sitting
exactly on the channel's extreme values, and filtering smooths those plateaus away.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal as sps

from .signals import Signal

QUALITY_THRESHOLDS = {"max_saturated": 0.001, "max_flat": 0.01, "max_mains": 0.10}


def _pct(fraction: float) -> str:
    """A fraction as a short percentage: 0.000333 -> '0.0333%', 0.25 -> '25%'."""
    return f"{100 * fraction:.3g}%"


def flat_fraction(x: np.ndarray, min_run: int = 5) -> float:
    """Fraction of samples that sit in a run of ``min_run`` or more identical consecutive values.

    Real recordings carry noise, so a run of identical values means a dropout, a stuck amplifier
    or an unplugged electrode.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    boundaries = np.concatenate(([0], np.flatnonzero(np.diff(x) != 0) + 1, [x.size]))
    lengths = np.diff(boundaries)
    return float(lengths[lengths >= min_run].sum() / x.size)


def saturated_fraction(
    x: np.ndarray, low: float | None = None, high: float | None = None, min_hits: int = 5
) -> float:
    """Fraction of samples sitting exactly on the converter's rails (clipping).

    A clean recording touches its extreme value once; a clipped one lands on it again and again.
    Each rail (``low`` and ``high``, by default the extremes of ``x``) counts when at least
    ``min_hits`` samples equal it, and then all of those samples are counted. Pass the extremes of
    the whole recording when ``x`` is only a piece of it, and ``min_hits=1`` together with the
    converter's real range when you know it. The result is 0 when ``low == high`` (a flat line is
    not clipping; :func:`flat_fraction` reports it).

    The rule suits fluctuating signals such as EMG and EEG. A trace that rests exactly on its
    minimum (a force sensor that reads 0 at rest) or a perfectly periodic synthetic wave repeats
    its extreme without being clipped, and will be reported.
    """
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    lo = finite.min() if low is None else low
    hi = finite.max() if high is None else high
    if hi == lo:
        return 0.0
    hits = 0
    for rail in (lo, hi):
        n = int(np.count_nonzero(x == rail))
        if n >= min_hits:
            hits += n
    return hits / x.size


def _mains_excess(
    x: np.ndarray,
    fs: float,
    mains: tuple[float, ...],
    max_harmonics: int,
    low_hz: float,
    half_width_hz: float,
    flank_hz: tuple[float, float],
) -> tuple[float, float]:
    """``(mains_hz, fraction)``: power above the neighbouring spectrum at the mains lines."""
    nperseg = int(min(x.size, max(64, round(2 * fs))))  # about 0.5 Hz resolution
    f, pxx = sps.welch(x, fs=fs, nperseg=nperseg)
    total = pxx[f >= low_hz].sum()
    if not np.isfinite(total) or total <= 0:
        return float("nan"), 0.0
    inner, outer = flank_hz
    best_hz, best = float("nan"), 0.0
    for base in mains:
        excess = 0.0
        for k in range(1, max_harmonics + 1):
            fc = base * k
            if fc + outer >= fs / 2:
                break
            near = np.abs(f - fc) <= half_width_hz
            flank = ((f >= fc - outer) & (f <= fc - inner)) | (
                (f >= fc + inner) & (f <= fc + outer)
            )
            if not near.any() or not flank.any():
                continue
            floor = np.median(pxx[flank])
            excess += float(np.clip(pxx[near] - floor, 0.0, None).sum())
        fraction = excess / total
        if fraction > best:
            best_hz, best = float(base), float(fraction)
    return best_hz, best


def signal_quality(
    sig: Signal,
    *,
    mains: tuple[float, ...] = (50.0, 60.0),
    max_harmonics: int = 5,
    min_hits: int = 5,
    rails: tuple[float, float] | None = None,
    flat_run: int = 5,
    max_saturated: float = QUALITY_THRESHOLDS["max_saturated"],
    max_flat: float = QUALITY_THRESHOLDS["max_flat"],
    max_mains: float = QUALITY_THRESHOLDS["max_mains"],
    meta: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """One row per channel: how much of the recording is unusable, and why.

    Columns
    -------
    ``nonfinite_fraction``
        Share of NaN / infinite samples.
    ``flat_fraction``
        Share of samples in runs of ``flat_run`` or more identical values (dropouts).
    ``saturated_fraction``
        Share of samples sitting exactly on the channel's extreme values, when at least
        ``min_hits`` do (see :func:`saturated_fraction`). ``rails=(low, high)`` replaces the
        extremes of the data with the converter's range, which you should pass if you know it
        (together with ``min_hits=1``).
    ``mains_hz``, ``mains_fraction``
        The mains frequency (from ``mains``) with the most interference, and the share of the
        channel's power (above 1 Hz) that the mains lines and their harmonics carry *in excess of
        the spectrum around them*. It is 0 for a clean channel and ``NaN`` in ``mains_hz`` when no
        line stands out. Computed with Welch's method; left ``NaN`` if the channel has
        non-finite samples.
    ``ok`` / ``issues``
        ``ok`` is False when any value is above its limit (``max_saturated``, ``max_flat``,
        ``max_mains``, and any non-finite sample); ``issues`` names the offenders.

    The limits are starting points, not standards: a recording that fails one is worth looking at,
    not necessarily worth discarding. ``meta`` columns (e.g. ``{"subject": "S01"}``) are prepended.
    """
    low, high = rails if rails is not None else (None, None)
    rows = []
    for name, x in zip(sig.channels, sig.data, strict=True):
        finite = np.isfinite(x)
        row: dict[str, Any] = {
            "channel": name,
            "n_samples": int(x.size),
            "nonfinite_fraction": float(1.0 - finite.mean()) if x.size else float("nan"),
            "flat_fraction": flat_fraction(x[finite], flat_run),
            "saturated_fraction": saturated_fraction(x, low, high, min_hits),
        }
        if finite.all() and x.size >= 8:
            row["mains_hz"], row["mains_fraction"] = _mains_excess(
                x, sig.fs, mains, max_harmonics, low_hz=1.0, half_width_hz=1.0, flank_hz=(3.0, 8.0)
            )
        else:
            row["mains_hz"], row["mains_fraction"] = float("nan"), float("nan")
        issues = []
        if row["nonfinite_fraction"] > 0:
            issues.append(f"{_pct(row['nonfinite_fraction'])} of samples are NaN or infinite")
        if row["flat_fraction"] > max_flat:
            issues.append(f"{_pct(row['flat_fraction'])} of samples in flat runs")
        if row["saturated_fraction"] > max_saturated:
            issues.append(f"{_pct(row['saturated_fraction'])} of samples clipped")
        if row["mains_fraction"] > max_mains:
            issues.append(
                f"mains {row['mains_hz']:g} Hz carries {_pct(row['mains_fraction'])} of power"
            )
        row["ok"] = not issues
        row["issues"] = "; ".join(issues)
        rows.append(row)
    table = pd.DataFrame(rows)
    for pos, (key, value) in enumerate((meta or {}).items()):
        table.insert(pos, key, value)
    return table
