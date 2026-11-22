"""Trial checks (with reasons), the force plateau and the way both reach the feature table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biosigkit import Signal, make_session, trial_feature_table
from biosigkit.force import plateau_window
from biosigkit.trials import check_trials, drop_invalid, plateau_events

FS = 100.0


def trapezoid(ramp_s, hold_s, level=10.0, rest_s=1.0, fs=FS):
    """Rest, ramp up, hold, ramp down, rest."""
    n = lambda seconds: round(seconds * fs)  # noqa: E731
    return np.concatenate(
        [
            np.zeros(n(rest_s)),
            np.linspace(0.0, level, n(ramp_s), endpoint=False),
            np.full(n(hold_s), level),
            np.linspace(level, 0.0, n(ramp_s), endpoint=False),
            np.zeros(n(rest_s)),
        ]
    )


@pytest.fixture
def recording():
    """Two channels, 30 s at 100 Hz, plain noise: every trial in the middle is fine."""
    rng = np.random.default_rng(7)
    return Signal(rng.standard_normal((2, 3000)), FS, ("emg1", "emg2"))


def events(*rows):
    return pd.DataFrame(rows, columns=["onset", "offset", "label"])


# ------------------------------------------------------------------------------ check_trials
def test_good_trials_are_valid_and_carry_their_extra_columns(recording):
    ev = events((2.0, 6.0, "a"), (10.0, 14.0, "b")).assign(level=[10, 30])
    report = check_trials(recording, ev)
    assert report["valid"].tolist() == [True, True]
    assert (report["reason"] == "").all() and (report["detail"] == "").all()
    assert report.columns.tolist() == [
        "trial", "onset", "offset", "label", "level", "valid", "reason", "detail"
    ]  # fmt: skip
    assert report["level"].tolist() == [10, 30]


def test_geometry_problems_get_their_own_reason(recording):
    ev = events(
        (-3.0, -1.0, "early"), (2.0, 6.0, "ok"), (10.0, 10.5, "short"), (28.0, 33.0, "late")
    )
    report = check_trials(recording, ev, trim_s=0.2).set_index("label")
    assert report.loc["ok", "valid"]
    assert report.loc["late", "reason"] == "outside_recording"
    assert report.loc["early", "reason"] == "outside_recording"
    assert report.loc["short", "reason"] == ""  # 0.5 - 0.4 = 0.1 s = 10 samples: long enough
    long_min = check_trials(recording, ev, trim_s=0.2, min_duration_s=0.5).set_index("label")
    assert long_min.loc["short", "reason"] == "too_short"
    assert "lies outside the recording" in long_min.loc["late", "detail"]


def test_rows_follow_the_sorted_events_and_trial_is_the_row_number(recording):
    ev = events((10.0, 14.0, "second"), (2.0, 6.0, "first"))
    report = check_trials(recording, ev)
    assert report["label"].tolist() == ["first", "second"]
    assert report["trial"].tolist() == [0, 1]


def test_quality_problems_are_found_with_their_channel_and_can_be_switched_off():
    rng = np.random.default_rng(11)
    data = rng.standard_normal((2, 3000))
    data[0, 300:320] = np.nan  # in trial "nan" (3-5 s)
    data[1, 700:760] = 0.5  # 60 held samples in trial "held" (7-9 s)
    rail = np.nanmax(np.abs(data[0])) + 1.0
    data[0, 1100:1300] = np.clip(
        5.0 * data[0, 1100:1300], -rail, rail
    )  # clipped, trial "clip" (11-13 s)
    sig = Signal(data, FS, ("emg1", "emg2"))
    ev = events((3.0, 5.0, "nan"), (7.0, 9.0, "held"), (11.0, 13.0, "clip"), (20.0, 22.0, "fine"))
    report = check_trials(sig, ev).set_index("label")
    assert report["reason"].to_dict() == {
        "nan": "nonfinite", "held": "flat", "clip": "saturated", "fine": ""
    }  # fmt: skip
    assert "emg1" in report.loc["nan", "detail"] and "emg2" in report.loc["held", "detail"]
    assert report.loc["fine", "valid"]
    assert check_trials(sig, ev, quality=False)["valid"].all()


def test_several_reasons_are_joined(recording):
    data = recording.data.copy()
    data[0, 500:520] = np.nan
    data[0, 520:560] = data[0, 520]  # flat as well
    report = check_trials(Signal(data, FS, recording.channels), events((4.5, 6.5, "x")))
    assert report.loc[0, "reason"] == "nonfinite;flat"


def test_a_plateau_column_marks_trials_without_a_plateau(recording):
    ev = events((2.0, 6.0, "a"), (10.0, 14.0, "b")).assign(plateau=[True, False])
    report = check_trials(recording, ev)
    assert report["reason"].tolist() == ["", "no_plateau"]


def test_drop_invalid_keeps_the_trial_numbers(recording):
    data = recording.data.copy()
    data[1, 1000:1010] = np.nan  # in the middle trial
    sig = Signal(data, FS, recording.channels)
    ev = events((2.0, 6.0, "a"), (18.0, 22.0, "c"), (10.0, 14.0, "b"))
    report = check_trials(sig, ev)
    assert report["valid"].tolist() == [True, False, True]
    kept = drop_invalid(ev, report)
    assert kept["label"].tolist() == ["a", "c"]
    assert kept["trial"].tolist() == [0, 2]  # the numbers of the report, not 0 and 1
    table = trial_feature_table(sig, kept)
    assert table["trial"].tolist() == [0, 2]
    with pytest.raises(ValueError, match="does not belong"):
        drop_invalid(ev.iloc[:2], report)


# ------------------------------------------------------------- trial_feature_table reports
def test_skipped_trials_are_listed_with_the_reason(recording):
    ev = events((2.0, 6.0, "a"), (28.0, 33.0, "late"), (-2.0, 1.0, "early"))
    table = trial_feature_table(recording, ev, on_invalid="skip")
    assert table["label"].tolist() == ["a"]
    rejected = table.attrs["rejected"]
    assert [(r["label"], r["reason"]) for r in rejected] == [
        ("early", "outside_recording"),
        ("late", "outside_recording"),
    ]
    assert trial_feature_table(recording, ev.iloc[[0]]).attrs["rejected"] == []


def test_the_error_says_what_is_wrong_with_the_trial(recording):
    with pytest.raises(ValueError, match=r"trial 1 .* lies outside the recording"):
        trial_feature_table(recording, events((2.0, 6.0, "a"), (28.0, 33.0, "b")))
    with pytest.raises(ValueError, match=r"is too short after trimming"):
        trial_feature_table(recording, events((2.0, 2.01, "a")))
    with pytest.raises(ValueError, match=r"no valid trials found \(outside_recording\)"):
        trial_feature_table(recording, events((28.0, 33.0, "b")), on_invalid="skip")


# ------------------------------------------------------------------------------ plateau
@pytest.mark.parametrize("ramp_s", [0.5, 1.0, 2.0])
def test_plateau_adapts_to_how_long_the_ramp_takes(ramp_s):
    trace = trapezoid(ramp_s, hold_s=3.0)
    start, stop = plateau_window(trace, FS)  # type: ignore[misc]
    # the force reaches 80 % of the level 80 % of the way up the ramp, and falls below it 20 % of
    # the way down: the window starts 20 % of a ramp early and ends 20 % of a ramp late
    assert start == pytest.approx(1.0 + 0.8 * ramp_s, abs=0.15)
    assert stop == pytest.approx(1.0 + ramp_s + 3.0 + 0.2 * ramp_s, abs=0.15)
    assert plateau_window(trace, FS, frac=0.95)[0] > start  # type: ignore[index]


def test_a_dip_during_the_hold_does_not_end_the_plateau():
    trace = trapezoid(0.5, hold_s=4.0)
    trace[300:310] *= 0.5  # a 0.1 s lapse in the middle of the hold
    start, stop = plateau_window(trace, FS)  # type: ignore[misc]
    assert stop - start > 4.0


def test_no_plateau_when_there_is_nothing_to_find():
    assert plateau_window(np.zeros(500), FS) is None  # never squeezed
    assert plateau_window(trapezoid(0.5, hold_s=0.2), FS, min_duration_s=1.0) is None
    assert plateau_window(np.array([1.0, 2.0]), FS) is None
    bad = trapezoid(0.5, 3.0)
    bad[200] = np.nan
    assert plateau_window(bad, FS) is None
    assert plateau_window(trapezoid(0.5, 3.0), FS, frac=0.9, min_duration_s=10.0) is None


def test_plateau_events_narrow_every_trial_of_a_simulated_session():
    session = make_session("S01", trials_per_level=2, seed=2)
    narrowed = plateau_events(session.events, session.force)
    assert narrowed["plateau"].all()
    assert (narrowed["onset"] > session.events["onset"]).all()
    assert (narrowed["offset"] < session.events["offset"]).all()
    assert narrowed["onset_trial"].tolist() == session.events["onset"].tolist()
    held = narrowed["offset"] - narrowed["onset"]
    assert held.median() == pytest.approx(5.2, abs=0.3)  # 6 s trials with 0.5 s ramps
    assert held.min() > 3.5 and held.max() < 5.6
    # the two windows that features can be taken from
    fixed = session.events["offset"] - session.events["onset"] - 2.0
    assert (held > fixed).all()
    assert check_trials(session.emg, narrowed)["valid"].all()


def test_trials_without_a_plateau_keep_their_window_and_are_reported():
    force = Signal(np.concatenate([trapezoid(0.5, 3.0), np.zeros(800)]), FS, ("force",))
    ev = events((0.0, 6.0, "squeezed"), (6.0, 14.0, "rest"))
    narrowed = plateau_events(ev, force)
    assert narrowed["plateau"].tolist() == [True, False]
    assert narrowed.loc[1, ["onset", "offset"]].tolist() == [6.0, 14.0]  # unchanged
    emg = Signal(np.random.default_rng(0).standard_normal((1, 1400)), FS, ("emg1",))
    assert check_trials(emg, narrowed)["reason"].tolist() == ["", "no_plateau"]


def test_plateau_events_pick_a_named_channel_and_cope_with_events_off_the_trace():
    trace = trapezoid(0.5, 3.0)
    force = Signal(np.vstack([np.zeros_like(trace), trace]), FS, ("a", "b"))
    ev = events((0.0, 6.0, "x"), (500.0, 506.0, "beyond the recording"))
    narrowed = plateau_events(ev, force, channel="b")
    assert narrowed["plateau"].tolist() == [True, False]
