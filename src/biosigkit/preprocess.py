"""Configurable preprocessing chain with automatic mains-interference detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import signal as sps

from .filters import bandpass, detrend, notch, resample
from .signals import Signal


def detect_line_noise(
    sig: Signal,
    *,
    mains: tuple[float, ...] = (50.0, 60.0),
    max_harmonics: int = 5,
    threshold_db: float = 6.0,
    search_hz: float = 1.0,
    flank_hz: tuple[float, float] = (3.0, 8.0),
    quantile: float = 0.25,
    segment_s: float = 1.0,
) -> list[float]:
    """Find power-line interference (fundamental and harmonics) in the spectrum.

    The recording is cut into ``segment_s`` segments and, for every frequency bin, the
    ``quantile`` of the power over time is kept, i.e. the spectrum of the *quietest* stretches.
    Mains interference is stationary while EMG bursts come and go, so this keeps the line visible
    even when strong contractions raise the broadband level (a plain average would hide it).

    A candidate frequency ``fc`` is flagged when the peak within ``fc ± search_hz`` exceeds the
    median of the two flanking bands (``fc ± flank_hz``) by ``threshold_db``. The mains frequency
    (50 or 60 Hz) with the strongest total evidence wins, so a 60 Hz recording is not also
    "notched" at 50 Hz. A single channel is enough to trigger a detection.

    Returns the nominal frequencies to notch, e.g. ``[60.0, 120.0]`` (empty if none).
    """
    nyquist = sig.fs / 2
    nperseg = int(min(sig.n_samples, max(64, round(segment_s * sig.fs))))
    f, _, sxx = sps.spectrogram(
        sig.data,
        fs=sig.fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        mode="psd",
        axis=-1,
    )
    p = np.quantile(sxx, quantile, axis=-1)  # (n_channels, n_freqs)
    inner, outer = flank_hz
    best_score, best_freqs = 0.0, []
    for base in mains:
        found: list[float] = []
        score = 0.0
        for k in range(1, max_harmonics + 1):
            fc = base * k
            if fc + outer >= nyquist or fc - outer <= 0:
                continue
            near = np.abs(f - fc) <= search_hz
            flank = ((f >= fc - outer) & (f <= fc - inner)) | (
                (f >= fc + inner) & (f <= fc + outer)
            )
            if not near.any() or not flank.any():
                continue
            peak = p[:, near].max(axis=-1)
            floor = np.median(p[:, flank], axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                prominence_db = 10.0 * np.log10(peak / floor)
            prominence_db = prominence_db[np.isfinite(prominence_db)]
            if prominence_db.size and prominence_db.max() >= threshold_db:
                found.append(float(fc))
                score += float(prominence_db.max())
        if score > best_score:
            best_score, best_freqs = score, found
    return best_freqs


def auto_notch(sig: Signal, **detect_kwargs: Any) -> tuple[Signal, list[float]]:
    """Detect mains interference and notch it. Returns ``(filtered_signal, notched_freqs)``."""
    freqs = detect_line_noise(sig, **detect_kwargs)
    return (notch(sig, freqs) if freqs else sig), freqs


@dataclass(frozen=True)
class PreprocessConfig:
    """Declarative description of a preprocessing chain.

    Steps run in this order: detrend → notch → band-pass → resample. (Linear zero-phase filters
    commute, so the order only matters for ``notch="auto"``, which inspects the broadband
    spectrum first.)

    Attributes
    ----------
    detrend:
        ``"linear"``, ``"constant"`` or ``None``.
    notch:
        ``"auto"`` (detect mains + harmonics), ``None``, one frequency or a tuple of them (Hz).
    band:
        ``(low, high)`` band-pass edges in Hz, or ``None``.
    order:
        Butterworth order per band edge.
    resample_to:
        Target sampling rate in Hz, or ``None``.
    mains:
        Candidate mains frequencies used when ``notch="auto"``.
    """

    detrend: str | None = "linear"
    notch: str | float | tuple[float, ...] | None = "auto"
    band: tuple[float, float] | None = None
    order: int = 4
    resample_to: float | None = None
    mains: tuple[float, ...] = (50.0, 60.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EMG_PRESET = PreprocessConfig(band=(20.0, 450.0))
EEG_PRESET = PreprocessConfig(band=(1.0, 45.0))


def _notch_frequencies(sig: Signal, cfg: PreprocessConfig) -> list[float]:
    spec = cfg.notch
    if spec is None:
        return []
    if isinstance(spec, str):
        if spec != "auto":
            raise ValueError("notch must be 'auto', None or frequencies in Hz")
        return detect_line_noise(sig, mains=cfg.mains)
    if isinstance(spec, int | float):
        return [float(spec)]
    return [float(f) for f in spec]


def preprocess(sig: Signal, cfg: PreprocessConfig | None = None) -> Signal:
    """Apply a :class:`PreprocessConfig` (default: :data:`EMG_PRESET`).

    The returned signal records what was done under ``meta["preprocess"]`` so results stay
    traceable (the demo stores it in the database).
    """
    cfg = cfg or EMG_PRESET
    out = sig
    if cfg.detrend:
        out = detrend(out, cfg.detrend)
    notch_freqs = _notch_frequencies(out, cfg)
    if notch_freqs:
        out = notch(out, notch_freqs)
    if cfg.band is not None:
        out = bandpass(out, cfg.band[0], cfg.band[1], order=cfg.order)
    if cfg.resample_to is not None and cfg.resample_to != out.fs:
        out = resample(out, cfg.resample_to)
    provenance = {
        **cfg.to_dict(),
        "notch_applied_hz": notch_freqs,
        "fs_in": sig.fs,
        "fs_out": out.fs,
    }
    return out.with_meta(preprocess=provenance)
