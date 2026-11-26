"""SQLite round trips, constraints, parameterised queries and the bundled analytical SQL."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from biosigkit import connect, make_session, trial_feature_table, window_feature_table
from biosigkit.storage import (
    list_examples,
    load_feature_table,
    query,
    read_example,
    register_recording,
    save_feature_table,
)
from biosigkit.tables import infer_feature_columns

EXAMPLES = [
    "mean_rms_by_label",
    "within_subject_minmax",
    "activation_episodes",
    "top_trials_per_subject",
    "data_quality",
    "missing_values",
    "duplicate_recordings",
]


@pytest.fixture(scope="module")
def tables():
    """Window and trial tables of two simulated subjects (raw EMG: fast, no filtering)."""
    windows, trials = [], []
    for i, subject in enumerate(("S01", "S02")):
        s = make_session(subject, trials_per_level=2, seed=i)
        meta = {"subject": subject, "session": "S1"}
        windows.append(
            window_feature_table(s.emg, win_s=0.2, step_s=0.1, events=s.events, meta=meta)
        )
        trials.append(trial_feature_table(s.emg, s.events, trim_s=1.0, meta=meta))
    return pd.concat(windows, ignore_index=True), pd.concat(trials, ignore_index=True)


@pytest.fixture
def conn():
    """An empty in-memory database, closed after the test."""
    connection = connect()
    yield connection
    connection.close()


@pytest.fixture
def db(tables, conn):
    windows, trials = tables
    save_feature_table(conn, windows, kind="window")
    save_feature_table(conn, trials, kind="trial")
    for subject in ("S01", "S02"):  # the same samples under two names, for the duplicates query
        register_recording(conn, subject, "S1", content_sha256="ab" * 32)
    return conn


# --------------------------------------------------------------------------------- schema
def test_schema_is_created_and_foreign_keys_are_enforced(conn):
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"subjects", "recordings", "segments", "features"} <= names
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO segments (recording_id, kind, t_start, t_stop) VALUES (999, 'trial', 0, 1)"
        )


def test_check_constraints_reject_inconsistent_segments(conn):
    rid = register_recording(conn, "S01")
    insert = "INSERT INTO segments (recording_id, kind, t_start, t_stop) VALUES (?, ?, ?, ?)"
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(insert, (rid, "trial", 2.0, 1.0))  # stops before it starts
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(insert, (rid, "epoch", 0.0, 1.0))  # unknown kind


def test_connecting_twice_to_the_same_file_is_harmless(tmp_path):
    path = tmp_path / "x.sqlite"
    connect(path).close()
    connect(path).close()  # the schema uses IF NOT EXISTS
    assert path.exists()


# ---------------------------------------------------------------------------- recordings
def test_register_recording_is_idempotent_and_never_erases_provenance(conn):
    a = register_recording(conn, "S01", "S1", fs=1000.0, preprocess={"notch": [60.0]})
    assert register_recording(conn, "S01", "S1") == a
    assert register_recording(conn, "S01", "S2") != a

    register_recording(conn, "S01", "S1", fs=2000.0)  # only fs is replaced
    fs, provenance = conn.execute(
        "SELECT fs_hz, preprocess FROM recordings WHERE recording_id = ?", (a,)
    ).fetchone()
    assert fs == 2000.0
    assert json.loads(provenance) == {"notch": [60.0]}


# ----------------------------------------------------------------------------- round trip
def test_trial_table_survives_a_round_trip(db, tables):
    _, trials = tables
    back = load_feature_table(db, "trial")
    cols = infer_feature_columns(trials)
    assert len(back) == len(trials)
    assert set(infer_feature_columns(back)) == set(cols)

    key = ["subject", "t_start"]
    a = trials.sort_values(key).reset_index(drop=True)
    b = back.sort_values(key).reset_index(drop=True)
    assert b["subject"].tolist() == a["subject"].tolist()
    assert b["label"].tolist() == a["label"].tolist()
    np.testing.assert_allclose(b[cols].to_numpy(), a[cols].to_numpy(), rtol=1e-12)


def test_window_and_trial_segments_stay_separate(db, tables):
    windows, trials = tables
    assert len(load_feature_table(db, "window")) == len(windows)
    assert len(load_feature_table(db, "trial")) == len(trials)
    with closing(connect()) as empty:
        assert load_feature_table(empty, "trial").empty


def test_nan_is_stored_as_null_and_comes_back_as_nan(conn):
    df = pd.DataFrame(
        {
            "subject": ["S01"],
            "t_start": [0.0],
            "t_stop": [1.0],
            "label": ["a"],
            "emg1_mnf": [np.nan],
            "emg1_rms": [1.5],
        }
    )
    save_feature_table(conn, df, kind="trial")
    assert conn.execute("SELECT COUNT(*) FROM features WHERE value IS NULL").fetchone()[0] == 1
    back = load_feature_table(conn, "trial")
    assert np.isnan(back.loc[0, "emg1_mnf"])
    assert back.loc[0, "emg1_rms"] == 1.5


def test_save_feature_table_validates_its_input(conn):
    good = pd.DataFrame({"subject": ["S01"], "t_start": [0.0], "t_stop": [1.0], "emg1_rms": [1.0]})
    with pytest.raises(ValueError, match="kind"):
        save_feature_table(conn, good, kind="epoch")
    with pytest.raises(ValueError, match="missing columns"):
        save_feature_table(conn, good.drop(columns="subject"), kind="trial")
    with pytest.raises(ValueError, match="no feature columns"):
        save_feature_table(conn, good.drop(columns="emg1_rms"), kind="trial")


def test_a_failing_subject_leaves_no_partial_rows_behind(conn):
    df = pd.DataFrame(
        {
            "subject": ["S01", "S01", "S02", "S02"],
            "t_start": [0.0, 1.0, 0.0, 5.0],
            "t_stop": [1.0, 2.0, 1.0, 4.0],  # S02's second row stops before it starts
            "emg1_rms": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(sqlite3.IntegrityError):
        save_feature_table(conn, df, kind="trial")
    saved = conn.execute(
        "SELECT DISTINCT r.subject_id FROM segments AS s JOIN recordings AS r USING (recording_id)"
    ).fetchall()
    assert saved == [("S01",)]  # S01 was committed, S02 was rolled back as a whole
    assert conn.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 2


# ---------------------------------------------------------------------------------- queries
def test_query_parameters_are_data_not_sql(db):
    evil = "S01'; DROP TABLE subjects; --"
    assert query(db, "SELECT * FROM subjects WHERE subject_id = ?", (evil,)).empty
    assert len(query(db, "SELECT * FROM subjects")) == 2  # still there
    named = query(db, "SELECT COUNT(*) AS n FROM segments WHERE kind = :kind", {"kind": "trial"})
    assert named.loc[0, "n"] == 16


def test_bundled_examples_are_listed_in_teaching_order():
    assert list_examples() == EXAMPLES


def test_read_example_accepts_several_spellings_and_rejects_unknown_names():
    text = read_example("activation_episodes")
    assert "ROW_NUMBER" in text
    assert read_example("03_activation_episodes") == text
    assert read_example("03_activation_episodes.sql") == text
    with pytest.raises(KeyError, match="unknown example"):
        read_example("does_not_exist")


@pytest.mark.parametrize("name", EXAMPLES)
def test_every_bundled_query_runs_and_returns_rows(db, name):
    out = query(db, read_example(name))
    assert len(out) > 0 and len(out.columns) >= 3


def test_gaps_and_islands_recovers_exactly_one_episode_per_trial(db, tables):
    _, trials = tables
    episodes = query(db, read_example("activation_episodes"))
    assert len(episodes) == len(trials)
    assert episodes.groupby("subject_id").size().to_dict() == {"S01": 8, "S02": 8}
    assert sorted(episodes["label"]) == sorted(trials["label"])
    assert episodes["duration_s"].between(5.5, 7.0).all()  # 6 s trials +/- one window
    assert (episodes["episode_start"] * 1000 % 1).round(6).eq(0).all()  # rounded to ms


def test_data_quality_flags_imbalanced_subjects(conn):
    rows = [("S01", "a")] * 6 + [("S01", "b")] * 2 + [("S02", "a")] * 4 + [("S02", "b")] * 4
    df = pd.DataFrame(
        {
            "subject": [r[0] for r in rows],
            "label": [r[1] for r in rows],
            "t_start": np.arange(len(rows), dtype=float),
            "t_stop": np.arange(len(rows), dtype=float) + 0.5,
            "emg1_rms": 1.0,
        }
    )
    save_feature_table(conn, df, kind="trial")
    quality = query(conn, read_example("data_quality")).set_index("subject_id")
    assert quality.loc["S01", "balance_flag"] == "IMBALANCED"  # 2 < 0.5 * 6
    assert quality.loc["S02", "balance_flag"] == "ok"
