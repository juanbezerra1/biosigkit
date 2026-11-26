"""Signal-quality report: clipping, flat lines, non-finite samples and mains contamination."""

from __future__ import annotations

import numpy as np
import pytest

from biosigkit import Signal
from biosigkit.quality import flat_fraction, saturated_fraction, signal_quality
from conftest import sine

FS = 1000.0


def noise(n_channels=1, seconds=20.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_channels, round(FS * seconds)))


# ----------------------------------------------------------------------------- run helpers
def test_flat_fraction_counts_dropouts_but_not_noise():
    x = noise()[0]
    assert flat_fraction(x) == 0.0
    x[2000:2200] = x[2000]  # 200 samples held: an amplifier that froze
    assert flat_fraction(x) == pytest.approx(200 / x.size)
    assert flat_fraction(np.ones(50)) == 1.0
    assert np.isnan(flat_fraction(np.array([])))


def test_clipping_is_many_samples_on_the_extreme_not_one_peak():
    rng = np.random.default_rng(3)
    clean = 3.0 * sine(10.0, FS, 5.0) + 0.05 * rng.standard_normal(5000)
    assert saturated_fraction(clean) == 0.0  # each extreme is touched once
    clipped = np.clip(clean, -1.0, 1.0)
    assert saturated_fraction(clipped) == pytest.approx(np.mean(np.abs(clean) >= 1.0), abs=1e-9)
    spike = rng.standard_normal(5000)
    spike[10] = 100.0  # a lone artefact is not clipping
    assert saturated_fraction(spike) == 0.0
    assert saturated_fraction(np.ones(20)) == 0.0  # a flat line is not clipping
    assert np.isnan(saturated_fraction(np.array([np.nan, np.nan])))


def test_saturation_can_use_the_rails_of_the_whole_recording():
    x = np.concatenate([np.full(10, 5.0), np.zeros(10)])
    assert saturated_fraction(x[:10], low=0.0, high=5.0) == 1.0
    assert saturated_fraction(x[10:], low=0.0, high=5.0) == 1.0  # the low rail is 0
    assert saturated_fraction(x[10:]) == 0.0  # alone, the piece is just flat
    assert saturated_fraction(np.array([0.0, 1.0, 5.0, 2.0]), 0.0, 5.0, min_hits=1) == 0.5


# ------------------------------------------------------------------------------ the report
def test_a_clean_recording_passes_every_check():
    q = signal_quality(Signal(noise(2), FS, ("emg1", "emg2")))
    assert q["channel"].tolist() == ["emg1", "emg2"]
    assert q["ok"].all() and (q["issues"] == "").all()
    assert (q[["nonfinite_fraction", "flat_fraction", "saturated_fraction"]] == 0).all().all()
    assert (q["mains_fraction"] < 0.02).all()
    assert q["n_samples"].tolist() == [20000, 20000]


@pytest.mark.parametrize("line_hz", [50.0, 60.0])
def test_mains_share_of_power_is_measured_and_attributed_to_the_right_frequency(line_hz):
    x = noise(1)[0] + 2.0 * sine(line_hz, FS, 20.0)  # sine power 2, noise power 1: 2/3 of the total
    q = signal_quality(Signal(x, FS))
    assert q.loc[0, "mains_hz"] == line_hz
    assert q.loc[0, "mains_fraction"] == pytest.approx(2 / 3, abs=0.08)
    assert not q.loc[0, "ok"] and f"mains {line_hz:g} Hz" in q.loc[0, "issues"]


def test_only_the_damaged_channel_is_flagged():
    data = noise(3, seed=4)
    data[1, 500] = 30.0  # one spike is an artefact, not clipping
    data[2] = np.clip(4.0 * data[2], -3.0, 3.0)
    q = signal_quality(Signal(data, FS, ("a", "b", "c"))).set_index("channel")
    assert q["ok"].to_dict() == {"a": True, "b": True, "c": False}
    assert q.loc["c", "saturated_fraction"] > 0.1
    assert "clipped" in q.loc["c", "issues"]


def test_dropouts_and_non_finite_samples_are_reported():
    data = noise(2, seed=5)
    data[0, 5000:5300] = 0.25  # 300 samples held at a constant value
    data[1, 100] = np.nan
    q = signal_quality(Signal(data, FS, ("held", "gap"))).set_index("channel")
    assert q.loc["held", "flat_fraction"] == pytest.approx(300 / 20000)
    assert not q.loc["held", "ok"] and "flat runs" in q.loc["held", "issues"]
    assert q.loc["gap", "nonfinite_fraction"] == pytest.approx(1 / 20000)
    assert not q.loc["gap", "ok"] and "NaN or infinite" in q.loc["gap", "issues"]
    assert np.isnan(q.loc["gap", "mains_fraction"])  # not computed on a channel with holes


def test_min_hits_and_a_known_converter_range():
    x = noise(1, seed=8)[0]
    x[[10, 20, 30]] = x.max() + 1.0  # three samples on one extreme
    sig = Signal(x, FS)
    assert signal_quality(sig).loc[0, "saturated_fraction"] == 0.0  # below the default of 5 hits
    assert signal_quality(sig, min_hits=3).loc[0, "saturated_fraction"] == pytest.approx(3 / 20000)
    # with the real range of the converter (+-100) nothing sits on a rail
    assert (
        signal_quality(sig, rails=(-100.0, 100.0), min_hits=1).loc[0, "saturated_fraction"] == 0.0
    )


def test_limits_and_meta_columns_can_be_set():
    x = noise(1)[0] + 0.5 * sine(60.0, FS, 20.0)  # about 11 % of the power
    sig = Signal(x, FS)
    assert signal_quality(sig, max_mains=0.5).loc[0, "ok"]
    q = signal_quality(sig, max_mains=0.05, meta={"subject": "S01", "session": "S1"})
    assert list(q.columns[:2]) == ["subject", "session"] and not q.loc[0, "ok"]


def test_tiny_and_low_rate_signals_do_not_crash():
    assert signal_quality(Signal(np.arange(5.0), FS)).loc[0, "channel"] == "ch1"
    low = signal_quality(Signal(noise(1, seconds=5)[0], 100.0))  # Nyquist 50 Hz: no mains line fits
    assert low.loc[0, "mains_fraction"] == 0.0
