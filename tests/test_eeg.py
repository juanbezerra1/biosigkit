import numpy as np
import pytest

from biosigkit import Signal
from biosigkit.eeg import (
    BANDS,
    band_power,
    band_powers,
    common_average_reference,
    eeg_features,
    erd_percent,
    psd,
    reject_by_ptp,
)
from conftest import sine

FS = 250.0


def test_band_power_recovers_known_sine_power():
    x = sine(10, FS, 20, amp=2.0)  # power = A^2 / 2 = 2
    assert band_power(x, FS, BANDS["alpha"]) == pytest.approx(2.0, rel=0.05)
    powers = band_powers(x, FS)
    assert powers["alpha"] > 20 * max(powers["beta"], powers["theta"], powers["delta"])


def test_relative_power_and_psd_shapes():
    x = np.stack([sine(10, FS, 20), sine(20, FS, 20)])
    rel = band_power(x, FS, BANDS["alpha"], relative=True)
    assert rel.shape == (2,)
    assert rel[0] > 0.9 and rel[1] < 0.05
    f, p = psd(x, FS, nperseg_s=2.0)
    assert p.shape == (2, f.size)
    assert f[1] - f[0] == pytest.approx(0.5)


def test_band_too_narrow_for_resolution_raises():
    with pytest.raises(ValueError, match="fewer than 2"):
        band_power(sine(10, FS, 20), FS, (9.9, 10.1))


def test_erd_percent_sign_convention_and_zero_reference():
    out = erd_percent(np.array([1.0, 3.0, 5.0]), np.array([4.0, 2.0, 0.0]))
    np.testing.assert_allclose(out[:2], [-75.0, 50.0])
    assert np.isnan(out[2])


def test_halving_amplitude_gives_minus_75_percent_erd():
    rest = band_power(sine(10, FS, 20, amp=2.0), FS, BANDS["alpha"])
    task = band_power(sine(10, FS, 20, amp=1.0), FS, BANDS["alpha"])
    assert erd_percent(task, rest) == pytest.approx(-75.0, abs=3.0)


def test_common_average_reference_removes_shared_component():
    rng = np.random.default_rng(0)
    common = rng.normal(size=1000)
    own = rng.normal(size=(3, 1000))
    car = common_average_reference(Signal(own + common, FS))
    np.testing.assert_allclose(car.data.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(car.data, own - own.mean(axis=0), atol=1e-12)
    with pytest.raises(ValueError):
        common_average_reference(Signal(own[0], FS))


def test_reject_by_ptp_flags_large_artifacts():
    epochs = np.zeros((3, 2, 100))
    epochs[1, 0, 50] = 500.0  # blink-like spike on one channel
    epochs[2] = np.sin(np.linspace(0, 6, 100)) * 10
    np.testing.assert_array_equal(reject_by_ptp(epochs, 100.0), [True, False, True])


def test_eeg_features_has_one_array_per_band():
    seg = np.random.default_rng(1).normal(size=(4, 3, 250))
    out = eeg_features(seg, FS)
    assert list(out) == list(BANDS)
    assert all(v.shape == (4, 3) for v in out.values())
