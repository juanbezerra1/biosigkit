"""EEG helpers: spectral band power, re-referencing, ERD/ERS and epoch rejection."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import signal as sps
from scipy.integrate import trapezoid

from .signals import Signal

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def common_average_reference(sig: Signal) -> Signal:
    """Subtract the instantaneous mean across channels from every channel."""
    if sig.n_channels < 2:
        raise ValueError("common average reference needs at least 2 channels")
    return sig.with_data(sig.data - sig.data.mean(axis=0, keepdims=True))


def psd(
    x: np.ndarray, fs: float, *, nperseg_s: float = 2.0, overlap: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density along the last axis (``nperseg_s`` sets the resolution)."""
    x = np.asarray(x, dtype=float)
    nperseg = int(min(x.shape[-1], round(nperseg_s * fs)))
    return sps.welch(x, fs=fs, nperseg=nperseg, noverlap=int(nperseg * overlap), axis=-1)


def _integrate(f: np.ndarray, p: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    mask = (f >= band[0]) & (f <= band[1])
    if mask.sum() < 2:
        raise ValueError(
            f"band {band} Hz holds fewer than 2 PSD bins; use a longer segment or nperseg_s"
        )
    return trapezoid(p[..., mask], f[mask], axis=-1)


def band_power(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float],
    *,
    relative: bool = False,
    nperseg_s: float = 2.0,
    total_band: tuple[float, float] = (1.0, 45.0),
) -> np.ndarray:
    """Power in ``band`` (signal units²), optionally relative to ``total_band``."""
    f, p = psd(x, fs, nperseg_s=nperseg_s)
    power = _integrate(f, p, band)
    return power / _integrate(f, p, total_band) if relative else power


def band_powers(
    x: np.ndarray,
    fs: float,
    bands: Mapping[str, tuple[float, float]] = BANDS,
    *,
    relative: bool = False,
    nperseg_s: float = 2.0,
) -> dict[str, np.ndarray]:
    """:func:`band_power` for several bands, sharing one PSD estimate."""
    f, p = psd(x, fs, nperseg_s=nperseg_s)
    total = _integrate(f, p, (1.0, 45.0)) if relative else 1.0
    return {name: _integrate(f, p, band) / total for name, band in bands.items()}


def eeg_features(
    segments: np.ndarray,
    fs: float,
    bands: Mapping[str, tuple[float, float]] = BANDS,
    *,
    relative: bool = False,
    nperseg_s: float = 1.0,
) -> dict[str, np.ndarray]:
    """Band-power features for ``(n_segments, n_channels, n_samples)`` (table-builder friendly)."""
    return band_powers(segments, fs, bands, relative=relative, nperseg_s=nperseg_s)


def erd_percent(task_power: np.ndarray, reference_power: np.ndarray) -> np.ndarray:
    """Event-related (de)synchronisation in percent: ``(task - ref) / ref * 100``.

    Negative values indicate desynchronisation (ERD), positive values synchronisation (ERS).
    Entries with a non-positive reference are returned as ``NaN``.
    """
    task = np.asarray(task_power, dtype=float)
    ref = np.asarray(reference_power, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (task - ref) / ref * 100.0
    return np.where(ref > 0, out, np.nan)


def reject_by_ptp(epochs: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of epochs to *keep*: peak-to-peak amplitude <= ``threshold`` on all channels.

    ``epochs`` has shape ``(n_epochs, n_channels, n_times)``.
    """
    ptp = np.ptp(np.asarray(epochs, dtype=float), axis=-1).max(axis=-1)
    return ptp <= threshold
