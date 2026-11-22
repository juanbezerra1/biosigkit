"""Tidy feature tables: window/trial labelling, long/wide conversion, summaries, validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biosigkit import Signal, trial_feature_table, window_feature_table
from biosigkit.tables import (
    features_to_long,
    features_to_wide,
    infer_feature_columns,
    label_by_overlap,
    summarize,
    validate_table,
)

EVENTS = pd.DataFrame(
    {
        "onset": [2.0, 6.0],
        "offset": [4.0, 8.0],
        "label": ["10%", "30%"],
        "level": [10, 30],
    }
)


@pytest.fixture
def two_level_signal(rng):
    """10 s at 1 kHz, two channels: amplitude 0.1 at rest, 1.0 (ch 1) / 2.0 (ch 2) in events."""
    fs, n = 1000.0, 10_000
    t = np.arange(n) / fs
    envelope = np.full(n, 0.1)
    for onset, offset in zip(EVENTS["onset"], EVENTS["offset"], strict=True):
        envelope[(t >= onset) & (t < offset)] = 1.0
    data = rng.standard_normal((2, n)) * envelope * np.array([[1.0], [2.0]])
    return Signal(data, fs, ("emg1", "emg2"))


# ------------------------------------------------------------------------------- labelling
def test_label_by_overlap_uses_majority_overlap_and_falls_back_to_rest():
    t_start = np.array([0.0, 1.5, 1.6, 1.4, 3.5, 5.5, 4.0])
    labels = label_by_overlap(t_start, t_start + 1.0, EVENTS)
    # 1.5-2.5 and 3.5-4.5 sit exactly 50 % inside an event, 1.4-2.4 only 40 %
    assert list(labels) == ["rest", "10%", "10%", "rest", "10%", "30%", "rest"]


def test_label_by_overlap_options_and_empty_events():
    t_start = np.array([1.4, 1.5])
    strict = label_by_overlap(t_start, t_start + 1.0, EVENTS, min_overlap=0.9, rest_label="idle")
    assert list(strict) == ["idle", "idle"]
    none = label_by_overlap(t_start, t_start + 1.0, EVENTS.iloc[0:0])
    assert list(none) == ["rest", "rest"]


# --------------------------------------------------------------------------- window tables
def test_window_table_layout_labels_and_channel_scaling(two_level_signal):
    table = window_feature_table(
        two_level_signal, win_s=0.5, step_s=0.5, events=EVENTS, meta={"subject": "S01"}
    )
    assert list(table.columns[:4]) == ["subject", "t_start", "t_stop", "label"]
    assert {"emg1_rms", "emg2_rms", "emg1_mav", "emg2_zc"} <= set(table.columns)
    assert len(table) == 20  # 10 s / 0.5 s, no overlap

    inside_first_event = table[(table["t_start"] >= 2.0) & (table["t_stop"] <= 4.0)]
    assert set(inside_first_event["label"]) == {"10%"}
    assert set(table.loc[table["t_start"] < 1.9, "label"]) == {"rest"}

    active = table[table["label"] != "rest"]
    ratio = (active["emg2_rms"] / active["emg1_rms"]).mean()
    assert ratio == pytest.approx(2.0, rel=0.1)  # channel 2 has twice the amplitude


def test_window_table_without_events_has_no_label_column(two_level_signal):
    table = window_feature_table(two_level_signal, win_s=1.0, step_s=1.0)
    assert "label" not in table.columns


def test_window_table_accepts_a_custom_feature_function(two_level_signal):
    def peak(windows, fs):
        return {"peak": np.abs(windows).max(axis=-1)}

    table = window_feature_table(two_level_signal, win_s=1.0, step_s=1.0, feature_fn=peak)
    assert list(table.columns) == ["t_start", "t_stop", "emg1_peak", "emg2_peak"]


def test_feature_names_must_not_contain_the_separator(two_level_signal):
    def bad(windows, fs):
        return {"bad_name": windows.mean(axis=-1)}

    with pytest.raises(ValueError, match="must not contain"):
        window_feature_table(two_level_signal, feature_fn=bad)


# ---------------------------------------------------------------------------- trial tables
def test_trial_table_has_one_row_per_event_and_keeps_extra_columns(two_level_signal):
    table = trial_feature_table(two_level_signal, EVENTS, meta={"subject": "S01", "session": "S1"})
    assert len(table) == 2
    assert list(table.columns[:2]) == ["subject", "session"]
    assert table["label"].tolist() == ["10%", "30%"]
    assert table["level"].tolist() == [10, 30]  # extra event column is carried through
    assert table["t_start"].tolist() == pytest.approx([2.0, 6.0])
    assert table["emg1_rms"].tolist() == pytest.approx([1.0, 1.0], rel=0.1)


def test_trial_trim_drops_the_edges(two_level_signal):
    table = trial_feature_table(two_level_signal, EVENTS, trim_s=0.5)
    assert table["t_start"].tolist() == pytest.approx([2.5, 6.5])
    assert table["t_stop"].tolist() == pytest.approx([3.5, 7.5])


def test_trials_outside_the_recording_raise_or_are_skipped(two_level_signal):
    events = pd.DataFrame({"onset": [2.0, 9.5], "offset": [4.0, 12.0], "label": ["a", "b"]})
    with pytest.raises(ValueError, match="outside the recording"):
        trial_feature_table(two_level_signal, events)
    skipped = trial_feature_table(two_level_signal, events, on_invalid="skip")
    assert skipped["label"].tolist() == ["a"]
    with pytest.raises(ValueError, match="no valid trials"):
        trial_feature_table(two_level_signal, events.iloc[1:], on_invalid="skip")


# ------------------------------------------------------------ columns, long/wide, summaries
def test_infer_feature_columns_skips_bookkeeping_and_non_numeric(two_level_signal):
    table = trial_feature_table(two_level_signal, EVENTS, meta={"subject": "S01"})
    table["note"] = ["x", "y"]
    cols = infer_feature_columns(table)
    assert "emg1_rms" in cols and "emg2_mnf" not in cols  # mnf is not a default feature
    bookkeeping = {"subject", "trial", "t_start", "t_stop", "label", "level", "note"}
    assert not bookkeeping & set(cols)


def test_long_and_wide_are_inverse_operations(two_level_signal):
    table = window_feature_table(two_level_signal, win_s=1.0, step_s=1.0)
    table["segment_id"] = np.arange(len(table))
    cols = infer_feature_columns(table)

    long = features_to_long(table, ["segment_id"], cols)
    assert set(long.columns) == {"segment_id", "value", "channel", "feature"}
    assert len(long) == len(table) * len(cols)
    assert set(long["channel"]) == {"emg1", "emg2"}

    wide = features_to_wide(long)
    pd.testing.assert_frame_equal(wide, table.set_index("segment_id")[cols])


def test_long_format_needs_channel_feature_column_names():
    df = pd.DataFrame({"id": [1, 2], "rms": [0.1, 0.2]})
    with pytest.raises(ValueError, match="channel"):
        features_to_long(df, ["id"], ["rms"])


def test_summarize_uses_flat_column_names():
    df = pd.DataFrame(
        {"label": ["a", "a", "b", "b"], "subject": ["s"] * 4, "emg1_rms": [1.0, 3.0, 5.0, 7.0]}
    )
    out = summarize(df, by=["label"])
    assert list(out.columns) == ["label", "emg1_rms.mean", "emg1_rms.std", "emg1_rms.count"]
    by_label = out.set_index("label")
    assert by_label.loc["a", "emg1_rms.mean"] == 2.0
    assert by_label.loc["b", "emg1_rms.count"] == 2


def test_validate_table_reports_missing_columns_and_non_finite_values():
    df = pd.DataFrame({"label": ["a"], "emg1_rms": [np.nan]})
    with pytest.raises(ValueError, match="missing required"):
        validate_table(df, required=["subject"])
    with pytest.raises(ValueError, match=r"non-finite.*emg1_rms"):
        validate_table(df)
    validate_table(df.assign(emg1_rms=1.0), required=["label"])
