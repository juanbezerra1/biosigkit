import numpy as np
import pytest

from biosigkit import EMG_PRESET, preprocess
from biosigkit.eeg import BANDS, band_power, erd_percent
from biosigkit.emg import rms
from biosigkit.synthetic import (
    bandlimited_noise,
    emg_from_force,
    force_profile,
    make_cohort,
    make_eeg,
    make_session,
    pink_noise,
)
from biosigkit.windowing import make_epochs


def test_bandlimited_noise_is_unit_variance_and_in_band(rng):
    x = bandlimited_noise(20000, 1000, (20, 200), rng)[0]
    assert x.std() == pytest.approx(1.0)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, 1 / 1000)
    assert spectrum[(freqs >= 15) & (freqs <= 250)].sum() / spectrum.sum() > 0.95


def test_pink_noise_has_more_low_than_high_frequency_power(rng):
    x = pink_noise(20000, rng)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    assert spectrum[10:100].mean() > 10 * spectrum[-1000:].mean()


def test_force_profile_is_a_trapezoid():
    prof = force_profile(10.0, 100, [(2.0, 8.0, 0.5)], ramp_s=1.0)
    assert prof.shape == (1000,)
    assert (
        prof[100] == 0
        and prof[500] == pytest.approx(0.5)
        and prof[950] == pytest.approx(0.0, abs=0.05)
    )
    assert prof[250] == pytest.approx(0.25, abs=0.01)  # halfway up the ramp


def test_emg_amplitude_follows_force(rng):
    force = force_profile(20.0, 100, [(2, 8, 0.2), (11, 17, 0.6)], ramp_s=0.5)
    emg = emg_from_force(force, 100, 1000, channel_gains=[1.0, 0.5], noise_std=0.0, rng=rng)
    assert emg.shape == (2, 20000)
    low, high = rms(emg[0, 4000:7000]), rms(emg[0, 12000:16000])
    assert low == pytest.approx(0.2, rel=0.1) and high == pytest.approx(0.6, rel=0.1)
    assert rms(emg[1, 12000:16000]) == pytest.approx(0.3, rel=0.1)


def test_make_session_structure_and_reproducibility():
    a = make_session("S01", trials_per_level=2, seed=7)
    b = make_session("S01", trials_per_level=2, seed=7)
    c = make_session("S01", trials_per_level=2, seed=8)
    assert a.emg.channels == ("emg1", "emg2") and a.emg.fs == 1000 and a.force.fs == 100
    assert len(a.events) == 8
    assert sorted(a.events["level"].unique()) == [10, 30, 50, 70]
    assert set(a.events["label"]) == {"10%", "30%", "50%", "70%"}
    np.testing.assert_array_equal(a.emg.data, b.emg.data)
    assert not np.array_equal(a.emg.data, c.emg.data)
    assert a.emg.duration == pytest.approx(a.force.duration, abs=0.01)


def test_session_force_tracks_the_target_and_mains_is_detectable():
    s = make_session("S01", trials_per_level=2, seed=3)
    ev = s.events.iloc[0]
    sl = slice(round((ev.onset + 1) * 100), round((ev.offset - 1) * 100))
    hold_target = s.target.data[0, sl]
    assert hold_target.std() < 1e-9
    assert abs(s.force.data[0, sl].mean() - hold_target.mean()) < 0.3 * hold_target.mean()
    assert 60.0 in preprocess(s.emg, EMG_PRESET).meta["preprocess"]["notch_applied_hz"]
    quiet = make_session("S01", trials_per_level=2, seed=3, line_noise=False)
    assert preprocess(quiet.emg, EMG_PRESET).meta["preprocess"]["notch_applied_hz"] == []


def test_cohort_has_distinct_subjects_with_variable_gain():
    cohort = make_cohort(6, seed=1, trials_per_level=1)
    assert [s.subject for s in cohort] == [f"S0{i}" for i in range(1, 7)]
    gains = [s.params["gain_mv"] for s in cohort]
    assert max(gains) / min(gains) > 1.3


def test_make_eeg_shows_stronger_erd_where_requested():
    sig, events = make_eeg(n_trials=12, seed=5)
    assert list(events["label"][:4]) == ["rest", "task", "rest", "task"]
    assert sig.channels == ("C3", "Cz", "C4") and sig.duration == pytest.approx(72.0)
    erd = {}
    for label in ("rest", "task"):
        ep = make_epochs(sig, events[events["label"] == label], tmin=0.5, tmax=2.5)
        erd[label] = band_power(ep.data, sig.fs, BANDS["alpha"], nperseg_s=1.0).mean(axis=0)
    values = erd_percent(erd["task"], erd["rest"])
    assert values[0] < -30 and values[0] < values[2]  # C3 (deep ERD) vs C4 (shallow)
