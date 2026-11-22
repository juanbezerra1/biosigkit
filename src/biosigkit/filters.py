"""Zero-phase digital filters, detrending and resampling for :class:`~biosigkit.Signal`."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import numpy as np
from scipy import signal as sps

from .signals import Signal


def _require_finite(sig: Signal) -> None:
    if not np.isfinite(sig.data).all():
        raise ValueError("signal contains NaN/inf; clean or interpolate it before filtering")


def _check_freqs(fs: float, *freqs: float) -> None:
    nyquist = fs / 2
    for f in freqs:
        if not 0 < f < nyquist:
            raise ValueError(f"frequency {f} Hz must be in (0, {nyquist}) Hz for fs={fs} Hz")


def _butter(sig: Signal, wn: float | list[float], btype: str, order: int) -> Signal:
    _require_finite(sig)
    sos = sps.butter(order, wn, btype=btype, fs=sig.fs, output="sos")
    return sig.with_data(sps.sosfiltfilt(sos, sig.data, axis=-1))


def detrend(sig: Signal, kind: str = "linear") -> Signal:
    """Remove a linear trend (``"linear"``) or the mean (``"constant"``) from every channel."""
    if kind not in {"linear", "constant"}:
        raise ValueError("kind must be 'linear' or 'constant'")
    _require_finite(sig)
    return sig.with_data(sps.detrend(sig.data, axis=-1, type=kind))


def bandpass(sig: Signal, low: float, high: float, order: int = 4) -> Signal:
    """Zero-phase Butterworth band-pass (``order`` per band edge, applied forward and backward)."""
    _check_freqs(sig.fs, low, high)
    if low >= high:
        raise ValueError("low cutoff must be smaller than high cutoff")
    return _butter(sig, [low, high], "bandpass", order)


def lowpass(sig: Signal, cutoff: float, order: int = 4) -> Signal:
    """Zero-phase Butterworth low-pass."""
    _check_freqs(sig.fs, cutoff)
    return _butter(sig, cutoff, "lowpass", order)


def highpass(sig: Signal, cutoff: float, order: int = 4) -> Signal:
    """Zero-phase Butterworth high-pass."""
    _check_freqs(sig.fs, cutoff)
    return _butter(sig, cutoff, "highpass", order)


def notch(sig: Signal, freqs: float | Sequence[float], q: float = 30.0) -> Signal:
    """Zero-phase IIR notch at one or several frequencies (e.g. mains and its harmonics)."""
    _require_finite(sig)
    if isinstance(freqs, int | float | np.floating):
        freq_list = [float(freqs)]
    else:
        freq_list = [float(f) for f in freqs]
    data = sig.data
    for f0 in freq_list:
        _check_freqs(sig.fs, f0)
        b, a = sps.iirnotch(f0, q, fs=sig.fs)
        data = sps.filtfilt(b, a, data, axis=-1)
    return sig.with_data(data)


def resample(sig: Signal, new_fs: float, max_denominator: int = 1000) -> Signal:
    """Polyphase resampling with anti-aliasing to ``new_fs`` (must be a rational ratio)."""
    _require_finite(sig)
    if new_fs <= 0:
        raise ValueError("new_fs must be positive")
    ratio = Fraction(new_fs / sig.fs).limit_denominator(max_denominator)
    if ratio == 0 or abs(float(ratio) * sig.fs - new_fs) > 1e-6 * new_fs:
        raise ValueError(f"cannot resample {sig.fs} Hz to {new_fs} Hz with a simple rational ratio")
    data = sps.resample_poly(sig.data, ratio.numerator, ratio.denominator, axis=-1)
    return sig.with_data(data, fs=sig.fs * float(ratio))
