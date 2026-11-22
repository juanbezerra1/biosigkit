import numpy as np
import pytest

from biosigkit import Signal
from biosigkit.emg import (
    ALL_FEATURES,
    detect_activations,
    emg_features,
    linear_envelope,
    moving_rms,
    normalize_mvc,
    spectral_moments,
    zero_crossings,
)
from conftest import sine

FS = 1000.0


def test_time_domain_features_match_analytic_values_for_a_sine():
    x = sine(50, FS, 1.0, amp=1.0, phase=0.3)[None, None, :]  # 50 whole cycles
    f = emg_features(x, FS, ("mav", "rms", "wl", "zc", "ssc", "var", "iemg"))
    assert f["mav"][0, 0] == pytest.approx(2 / np.pi, rel=0.01)
    assert f["rms"][0, 0] == pytest.approx(1 / np.sqrt(2), rel=0.01)
    assert f["var"][0, 0] == pytest.approx(0.5, rel=0.01)
    assert f["iemg"][0, 0] == pytest.approx(1000 * 2 / np.pi, rel=0.01)
    assert f["wl"][0, 0] == pytest.approx(200, rel=0.03)  # 4 * amplitude per cycle
    assert abs(f["zc"][0, 0] - 100) <= 1
    assert abs(f["ssc"][0, 0] - 100) <= 1


def test_features_are_vectorised_over_segments_and_channels():
    x = np.random.default_rng(0).normal(size=(5, 2, 200))
    out = emg_features(x, FS, ALL_FEATURES)
    assert set(out) == set(ALL_FEATURES)
    assert all(v.shape == (5, 2) for v in out.values())
    single = emg_features(x[3:4, 1:2], FS, ("rms",))["rms"][0, 0]
    assert single == pytest.approx(out["rms"][3, 1])


def test_zero_crossing_threshold_ignores_small_noise():
    noise = np.random.default_rng(1).normal(scale=0.01, size=1000)
    assert zero_crossings(noise) > 100
    assert zero_crossings(noise, threshold=0.5) == 0


def test_spectral_moments_locate_a_narrowband_signal():
    x = sine(100, FS, 2.0)
    mnf, mdf = spectral_moments(x, FS)
    assert mnf == pytest.approx(100, abs=5)
    assert mdf == pytest.approx(100, abs=5)
    mnf0, mdf0 = spectral_moments(np.zeros(500), FS)
    assert np.isnan(mnf0) and np.isnan(mdf0)


def test_unknown_feature_is_rejected():
    with pytest.raises(ValueError, match="unknown EMG feature"):
        emg_features(np.zeros((1, 1, 50)), FS, ("rms", "banana"))


def test_envelopes_follow_amplitude():
    rng = np.random.default_rng(2)
    x = rng.normal(scale=0.1, size=4000)
    x[1000:2000] *= 10
    sig = Signal(x, FS)
    for env in (linear_envelope(sig, 5.0), moving_rms(sig, 0.1)):
        assert env.data[0, 1200:1800].mean() > 5 * env.data[0, 2500:3500].mean()
        assert (env.data >= 0).all()
    steady = moving_rms(Signal(sine(50, FS, 2.0, amp=2.0), FS), 0.1)
    assert steady.data[0, 500:1500].mean() == pytest.approx(2 / np.sqrt(2), rel=0.02)


def test_normalize_mvc_scales_per_channel():
    sig = Signal(np.array([[1.0, 2.0], [3.0, 6.0]]), FS)
    out = normalize_mvc(sig, [2.0, 3.0])
    np.testing.assert_allclose(out.data, [[50.0, 100.0], [100.0, 200.0]])
    with pytest.raises(ValueError):
        normalize_mvc(sig, 0.0)


def _burst_signal(bursts, seconds=6.0, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.05, size=round(seconds * FS))
    for start, stop in bursts:
        x[round(start * FS) : round(stop * FS)] += rng.normal(
            scale=1.0, size=round((stop - start) * FS)
        )
    return linear_envelope(Signal(x, FS), 10.0).data[0]


def test_detect_activations_finds_bursts_at_the_right_times():
    env = _burst_signal([(1.5, 2.5), (4.0, 5.0)])
    acts = detect_activations(env, FS, baseline=(0.2, 1.0), k=3.0)
    assert len(acts) == 2
    np.testing.assert_allclose(acts["onset"], [1.5, 4.0], atol=0.1)
    np.testing.assert_allclose(acts["offset"], [2.5, 5.0], atol=0.1)
    assert (acts["peak"] > 0.3).all()


def test_detect_activations_merges_close_bursts_and_drops_blips():
    env = 0.1 + 0.01 * np.random.default_rng(4).random(6000)
    env[1500:2000] = 1.0
    env[2030:2500] = 1.0  # 30 ms gap
    env[4000:4010] = 1.0  # 10 ms blip
    merged = detect_activations(env, FS, baseline=(0.2, 1.0), min_gap_s=0.1, min_duration_s=0.05)
    assert len(merged) == 1
    assert merged["onset"].iloc[0] == pytest.approx(1.5)
    assert merged["offset"].iloc[0] == pytest.approx(2.5)
    split = detect_activations(env, FS, baseline=(0.2, 1.0), min_gap_s=0.01, min_duration_s=0.05)
    assert len(split) == 2
    kept_blip = detect_activations(
        env, FS, baseline=(0.2, 1.0), min_gap_s=0.1, min_duration_s=0.005
    )
    assert len(kept_blip) == 2
    assert detect_activations(np.full(2000, 0.01), FS, baseline=(0.0, 1.0)).empty


def test_detect_activations_validates_input():
    with pytest.raises(ValueError, match="baseline"):
        detect_activations(np.zeros(100), FS, baseline=(0.0, 0.0005))
    with pytest.raises(ValueError, match="1-D"):
        detect_activations(np.zeros((2, 100)), FS, baseline=(0.0, 0.05))
