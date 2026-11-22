import pytest

from biosigkit import EEG_PRESET, EMG_PRESET, PreprocessConfig, Signal, make_session, preprocess
from biosigkit.preprocess import auto_notch, detect_line_noise
from conftest import amplitude_at, sine

FS = 1000.0


def _noise_with_lines(rng, freqs_amps, seconds=20):
    x = rng.normal(size=round(FS * seconds))
    for f, a in freqs_amps:
        x = x + sine(f, FS, seconds, amp=a)
    return Signal(x, FS)


def test_detects_mains_and_harmonic(rng):
    assert detect_line_noise(_noise_with_lines(rng, [(60, 1.0), (120, 0.5)])) == [60.0, 120.0]


def test_distinguishes_50_from_60_hz(rng):
    assert detect_line_noise(_noise_with_lines(rng, [(50, 1.0)])) == [50.0]


def test_clean_noise_triggers_nothing(rng):
    assert detect_line_noise(_noise_with_lines(rng, [])) == []


def test_auto_notch_attenuates_detected_lines(rng):
    sig = _noise_with_lines(rng, [(60, 1.0), (120, 0.5)])
    out, freqs = auto_notch(sig)
    assert freqs == [60.0, 120.0]
    assert amplitude_at(out.data[0], FS, 60) < 0.05
    assert amplitude_at(sig.data[0], FS, 60) > 0.9  # the input was not modified


def test_preprocess_records_provenance_and_resamples(rng):
    sig = _noise_with_lines(rng, [(60, 1.0), (120, 0.5)], seconds=10)
    cfg = PreprocessConfig(band=(20.0, 250.0), resample_to=600.0)
    out = preprocess(sig, cfg)
    assert out.fs == pytest.approx(600.0)
    assert out.n_samples == 6000
    info = out.meta["preprocess"]
    assert info["notch_applied_hz"] == [60.0, 120.0]
    assert info["fs_in"] == FS and info["fs_out"] == pytest.approx(600.0)
    assert "preprocess" not in sig.meta


def test_presets_and_explicit_notch(rng):
    sig = _noise_with_lines(rng, [(60, 1.0)], seconds=10)
    assert preprocess(sig, EMG_PRESET).meta["preprocess"]["band"] == (20.0, 450.0)
    assert preprocess(sig, EEG_PRESET).meta["preprocess"]["band"] == (1.0, 45.0)
    fixed = preprocess(sig, PreprocessConfig(notch=60.0, band=None, detrend=None))
    assert fixed.meta["preprocess"]["notch_applied_hz"] == [60.0]
    off = preprocess(sig, PreprocessConfig(notch=None, band=None))
    assert off.meta["preprocess"]["notch_applied_hz"] == []


def test_invalid_notch_spec_raises(rng):
    with pytest.raises(ValueError, match="notch"):
        preprocess(_noise_with_lines(rng, []), PreprocessConfig(notch="sometimes"))


def test_mains_detection_is_reliable_on_simulated_emg():
    """Strong contractions must not hide the line, and a clean recording must not trigger it."""
    for seed in range(15):
        with_line = make_session("S", trials_per_level=3, seed=seed)
        clean = make_session("S", trials_per_level=3, seed=seed, line_noise=False)
        assert detect_line_noise(with_line.emg) == [60.0, 120.0], f"missed the line, seed {seed}"
        assert detect_line_noise(clean.emg) == [], f"false alarm, seed {seed}"
