"""Surface-EMG processing: envelopes, classic time/frequency features and onset detection."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy import signal as sps

from .filters import lowpass
from .signals import Signal

DEFAULT_FEATURES = ("mav", "rms", "wl", "zc", "ssc")
ALL_FEATURES = ("mav", "rms", "wl", "zc", "ssc", "var", "iemg", "mnf", "mdf")


# --------------------------------------------------------------------------- envelopes
def linear_envelope(sig: Signal, cutoff: float = 5.0, order: int = 4) -> Signal:
    """Full-wave rectification followed by a zero-phase low-pass (the "linear envelope")."""
    env = lowpass(sig.with_data(np.abs(sig.data)), cutoff, order=order)
    return env.with_data(np.clip(env.data, 0.0, None))


def moving_rms(sig: Signal, window_s: float = 0.1) -> Signal:
    """Centred moving-RMS envelope."""
    size = max(1, round(window_s * sig.fs))
    power = ndimage.uniform_filter1d(np.square(sig.data), size=size, axis=-1, mode="nearest")
    return sig.with_data(np.sqrt(np.clip(power, 0.0, None)))


def normalize_mvc(sig: Signal, mvc: float | Sequence[float]) -> Signal:
    """Express an amplitude signal as % of a per-channel MVC reference value."""
    ref = np.broadcast_to(np.asarray(mvc, dtype=float), (sig.n_channels,)).copy()
    if not np.all(ref > 0):
        raise ValueError("MVC reference values must be positive")
    return sig.with_data(100.0 * sig.data / ref[:, None])


# --------------------------------------------------------------------------- features
# All feature functions work on the last axis, so they accept a single window ``(n,)`` as well as
# a batch ``(n_windows, n_channels, n)`` and return arrays with the last axis reduced.
def mav(x: np.ndarray) -> np.ndarray:
    """Mean absolute value."""
    return np.mean(np.abs(x), axis=-1)


def rms(x: np.ndarray) -> np.ndarray:
    """Root mean square."""
    return np.sqrt(np.mean(np.square(x), axis=-1))


def waveform_length(x: np.ndarray) -> np.ndarray:
    """Cumulative length of the waveform: ``sum(|x[i+1] - x[i]|)``."""
    return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1)


def integrated_emg(x: np.ndarray) -> np.ndarray:
    """Sum of absolute values."""
    return np.sum(np.abs(x), axis=-1)


def variance(x: np.ndarray) -> np.ndarray:
    return np.var(x, axis=-1)


def zero_crossings(x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Sign changes whose step is at least ``threshold`` (suppresses noise-induced crossings)."""
    crossing = x[..., :-1] * x[..., 1:] < 0
    big_enough = np.abs(np.diff(x, axis=-1)) >= threshold
    return np.sum(crossing & big_enough, axis=-1)


def slope_sign_changes(x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Local extrema where ``(x[i]-x[i-1]) * (x[i]-x[i+1])`` exceeds ``threshold``."""
    d1 = x[..., 1:-1] - x[..., :-2]
    d2 = x[..., 1:-1] - x[..., 2:]
    return np.sum(d1 * d2 > threshold, axis=-1)


def spectral_moments(x: np.ndarray, fs: float, nperseg: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Mean (MNF) and median (MDF) frequency from the Welch PSD. ``NaN`` for all-zero windows."""
    f, p = sps.welch(x, fs=fs, nperseg=min(x.shape[-1], nperseg), axis=-1)
    total = p.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mnf = (p * f).sum(axis=-1) / total
        reached = np.cumsum(p, axis=-1) >= (total / 2)[..., None]
    mdf = f[np.argmax(reached, axis=-1)]
    valid = total > 0
    return np.where(valid, mnf, np.nan), np.where(valid, mdf, np.nan)


def emg_features(
    segments: np.ndarray,
    fs: float,
    names: Sequence[str] = DEFAULT_FEATURES,
    *,
    zc_threshold: float = 0.0,
    ssc_threshold: float = 0.0,
    nperseg: int = 256,
) -> dict[str, np.ndarray]:
    """Compute named features for a batch of segments.

    Parameters
    ----------
    segments:
        Array ``(n_segments, n_channels, n_samples)``.
    names:
        Any of :data:`ALL_FEATURES`: ``mav, rms, wl, zc, ssc, var, iemg, mnf, mdf``. ``zc``,
        ``ssc``, ``wl``, ``mav`` and ``rms`` are the classic Hudgins-style time-domain set.

    Returns
    -------
    dict mapping feature name to an array ``(n_segments, n_channels)`` of floats.
    """
    x = np.asarray(segments, dtype=float)
    unknown = [n for n in names if n not in ALL_FEATURES]
    if unknown:
        raise ValueError(f"unknown EMG feature(s) {unknown}; choose from {ALL_FEATURES}")
    spectral = spectral_moments(x, fs, nperseg) if {"mnf", "mdf"} & set(names) else None
    out: dict[str, np.ndarray] = {}
    for name in names:
        if name == "mav":
            out[name] = mav(x)
        elif name == "rms":
            out[name] = rms(x)
        elif name == "wl":
            out[name] = waveform_length(x)
        elif name == "zc":
            out[name] = zero_crossings(x, zc_threshold).astype(float)
        elif name == "ssc":
            out[name] = slope_sign_changes(x, ssc_threshold).astype(float)
        elif name == "var":
            out[name] = variance(x)
        elif name == "iemg":
            out[name] = integrated_emg(x)
        elif spectral is not None:  # "mnf" / "mdf"
            out[name] = spectral[0] if name == "mnf" else spectral[1]
    return out


# --------------------------------------------------------------------------- onset detection
def baseline_threshold(
    env: np.ndarray, fs: float, baseline: tuple[float, float], k: float = 3.0
) -> float:
    """``mean + k * std`` of the envelope inside ``baseline = (start_s, stop_s)`` of ``env``."""
    start, stop = (round(b * fs) for b in baseline)
    ref = np.asarray(env, dtype=float)[start:stop]
    if ref.size < 2:
        raise ValueError("baseline window must contain at least 2 samples")
    return float(ref.mean() + k * ref.std(ddof=1))


def detect_activations(
    env: np.ndarray,
    fs: float,
    *,
    baseline: tuple[float, float],
    k: float = 3.0,
    min_duration_s: float = 0.05,
    min_gap_s: float = 0.05,
    t0: float = 0.0,
) -> pd.DataFrame:
    """Threshold-based detection of muscle activations in a 1-D envelope.

    The threshold is ``mean + k * std`` of a resting ``baseline`` window. Supra-threshold runs
    separated by less than ``min_gap_s`` are merged and runs shorter than ``min_duration_s`` are
    dropped. Returns one row per activation with ``onset``, ``offset``, ``duration`` (seconds,
    shifted by ``t0``) and the envelope ``peak``.
    """
    x = np.asarray(env, dtype=float)
    if x.ndim != 1:
        raise ValueError("env must be 1-D (a single channel)")
    threshold = baseline_threshold(x, fs, baseline, k)
    edges = np.diff(np.concatenate(([0], (x > threshold).astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    merged: list[list[int]] = []
    for s, e in zip(starts, stops, strict=True):
        if merged and (s - merged[-1][1]) / fs < min_gap_s:
            merged[-1][1] = int(e)
        else:
            merged.append([int(s), int(e)])
    rows = [
        {
            "onset": t0 + s / fs,
            "offset": t0 + e / fs,
            "duration": (e - s) / fs,
            "peak": float(x[s:e].max()),
        }
        for s, e in merged
        if (e - s) / fs >= min_duration_s
    ]
    return pd.DataFrame(rows, columns=["onset", "offset", "duration", "peak"])
