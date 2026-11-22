import numpy as np
import pytest

from biosigkit import Signal
from biosigkit.filters import bandpass, detrend, highpass, lowpass, notch, resample
from conftest import amplitude_at, sine

FS = 2000.0


def _three_tones():
    x = sine(10, FS, 10) + sine(100, FS, 10) + sine(400, FS, 10)
    return Signal(x, FS)


def test_bandpass_keeps_passband_and_removes_the_rest():
    y = bandpass(_three_tones(), 50, 200).data[0]
    assert amplitude_at(y, FS, 100) > 0.95
    assert amplitude_at(y, FS, 10) < 0.02
    assert amplitude_at(y, FS, 400) < 0.02


def test_filters_are_zero_phase():
    sig = Signal(sine(100, FS, 10), FS)
    y = bandpass(sig, 50, 200).data[0]
    mid = slice(2000, -2000)
    assert np.max(np.abs(y[mid] - sig.data[0][mid])) < 0.05


def test_lowpass_and_highpass_are_complementary():
    sig = _three_tones()
    lo = lowpass(sig, 50).data[0]
    hi = highpass(sig, 200).data[0]
    assert amplitude_at(lo, FS, 10) > 0.95 and amplitude_at(lo, FS, 400) < 0.02
    assert amplitude_at(hi, FS, 400) > 0.95 and amplitude_at(hi, FS, 10) < 0.02


def test_notch_removes_mains_but_keeps_neighbours():
    sig = Signal(sine(60, FS, 10) + sine(100, FS, 10) + sine(120, FS, 10), FS)
    y = notch(sig, [60, 120]).data[0]
    assert amplitude_at(y, FS, 60) < 0.05
    assert amplitude_at(y, FS, 120) < 0.05
    assert amplitude_at(y, FS, 100) > 0.95
    assert amplitude_at(notch(sig, 60).data[0], FS, 120) > 0.9  # scalar form only hits 60 Hz


@pytest.mark.parametrize(
    "call",
    [
        lambda s: bandpass(s, 100, 50),
        lambda s: bandpass(s, 10, FS / 2),
        lambda s: lowpass(s, 0),
        lambda s: notch(s, FS),
    ],
)
def test_invalid_frequencies_raise(call):
    with pytest.raises(ValueError):
        call(Signal(np.zeros(5000), FS))


def test_non_finite_values_are_rejected():
    x = sine(10, FS, 5)
    x[100] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        bandpass(Signal(x, FS), 5, 50)


def test_detrend_removes_linear_trend():
    t = np.arange(1000) / 100
    sig = Signal(3 * t + 5 + sine(2, 100, 10), 100)
    assert abs(np.polyfit(t, detrend(sig).data[0], 1)[0]) < 1e-3
    assert abs(detrend(sig, "constant").data.mean()) < 1e-9
    with pytest.raises(ValueError):
        detrend(sig, "quadratic")


def test_resample_changes_rate_and_preserves_tone():
    sig = Signal(sine(50, 1000, 10), 1000, t0=2.0)
    out = resample(sig, 600)
    assert out.fs == pytest.approx(600)
    assert out.n_samples == 6000
    assert out.t0 == 2.0
    assert amplitude_at(out.data[0], 600, 50) == pytest.approx(1.0, abs=0.02)


def test_resample_rejects_irrational_ratio():
    with pytest.raises(ValueError, match="rational"):
        resample(Signal(np.zeros(5000), 1000), 1000 * np.pi / 4, max_denominator=20)
