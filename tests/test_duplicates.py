"""Duplicate recordings and channels: hashing, grouping, and the SQLite column that stores it."""

from __future__ import annotations

import sqlite3
from contextlib import closing

import numpy as np

from biosigkit import Signal, connect, make_session
from biosigkit.duplicates import channel_hash, find_duplicates, signal_hash
from biosigkit.storage import list_examples, query, read_example, register_recording


def recording(seed, channels=("emg1", "emg2"), fs=1000.0, n=2000):
    rng = np.random.default_rng(seed)
    return Signal(rng.standard_normal((len(channels), n)), fs, channels)


# --------------------------------------------------------------------------------- hashing
def test_the_hash_depends_on_the_samples_and_the_rate_only():
    a = recording(1)
    assert signal_hash(a) == signal_hash(a.with_meta(subject="S9"))
    renamed = Signal(a.data.copy(), a.fs, ("x", "y"), t0=12.0, meta={"note": "copy"})
    assert signal_hash(renamed) == signal_hash(a)
    assert len(signal_hash(a)) == 64

    changed = a.data.copy()
    changed[1, 777] += 1e-9
    assert signal_hash(Signal(changed, a.fs, a.channels)) != signal_hash(a)
    assert signal_hash(Signal(a.data, 500.0, a.channels)) != signal_hash(a)
    assert signal_hash(a.pick(["emg2", "emg1"])) != signal_hash(a)  # channel order matters
    flat = Signal(a.data.reshape(1, -1), a.fs)  # same bytes, different shape
    assert signal_hash(flat) != signal_hash(a)


def test_channel_hash_matches_a_single_channel_recording():
    a = recording(2)
    assert channel_hash(a, "emg2") == signal_hash(a.pick(["emg2"]))
    assert channel_hash(a, "emg1") != channel_hash(a, "emg2")


# ------------------------------------------------------------------------------ the finder
def test_unique_recordings_give_an_empty_table_with_the_columns():
    found = find_duplicates({"a": recording(1), "b": recording(2)})
    assert found.empty and found.columns.tolist() == ["level", "n", "hash", "members"]


def test_a_copy_under_another_name_is_found_once():
    a = recording(1)
    copy = Signal(a.data.copy(), a.fs, ("p", "q"))  # even with other channel names
    found = find_duplicates({"S07/S1": a, "S12/S1": copy, "S03/S1": recording(3)})
    assert len(found) == 1  # the shared channels are not listed again
    row = found.iloc[0]
    assert (row["level"], row["n"], row["members"]) == ("recording", 2, "S07/S1, S12/S1")
    assert len(row["hash"]) == 12


def test_three_copies_form_one_group():
    a = recording(1)
    found = find_duplicates({n: Signal(a.data.copy(), a.fs) for n in ("x", "y", "z")})
    assert found["n"].tolist() == [3] and found.loc[0, "members"] == "x, y, z"


def test_a_channel_shared_by_different_recordings_is_found():
    a, b = recording(1), recording(2)
    b2 = Signal(np.vstack([b.data[0], a.data[1]]), b.fs, b.channels)  # b's emg2 is a's emg2
    found = find_duplicates({"a": a, "b": b2})
    assert found["level"].tolist() == ["channel"]
    assert found.loc[0, "members"] == "a/emg2, b/emg2"


def test_a_channel_repeated_inside_one_recording_is_found():
    a = recording(1)
    twin = Signal(np.vstack([a.data[0], a.data[0]]), a.fs, ("left", "right"))
    found = find_duplicates({"only": twin})
    assert found.loc[0, "level"] == "channel" and found.loc[0, "members"] == "only/left, only/right"


def test_two_unplugged_electrodes_are_not_duplicates_of_each_other():
    quiet = lambda seed: Signal(  # noqa: E731
        np.vstack([recording(seed).data[0], np.zeros(2000)]), 1000.0, ("emg1", "emg2")
    )
    assert find_duplicates({"a": quiet(1), "b": quiet(2)}).empty


def test_duplicates_in_a_simulated_cohort_are_only_the_ones_planted():
    sessions = {f"S{i}": make_session(f"S{i}", trials_per_level=1, seed=i).emg for i in range(3)}
    assert find_duplicates(sessions).empty
    sessions["S9"] = Signal(sessions["S1"].data.copy(), sessions["S1"].fs, ("e1", "e2"))
    assert find_duplicates(sessions)["members"].tolist() == ["S1, S9"]


# ------------------------------------------------------------------------------------ SQL
def test_register_recording_stores_the_hash_without_erasing_other_columns():
    with closing(connect()) as conn:
        a = register_recording(conn, "S01", "S1", fs=1000.0, content_sha256="ab" * 32)
        register_recording(conn, "S01", "S1", fs=2000.0)  # no hash given: keep the stored one
        fs, digest = conn.execute(
            "SELECT fs_hz, content_sha256 FROM recordings WHERE recording_id = ?", (a,)
        ).fetchone()
        assert (fs, digest) == (2000.0, "ab" * 32)


def test_the_sql_example_lists_hashes_shared_by_two_recordings():
    assert "duplicate_recordings" in list_examples()
    with closing(connect()) as conn:
        for subject, digest in (("S01", "aa" * 32), ("S02", "aa" * 32), ("S03", "bb" * 32)):
            register_recording(conn, subject, "S1", content_sha256=digest)
        register_recording(conn, "S04", "S1")  # never hashed: ignored, not a "duplicate of NULL"
        register_recording(conn, "S05", "S1")
        out = query(conn, read_example("duplicate_recordings"))
    assert out["n_recordings"].tolist() == [2]
    assert out.loc[0, "recordings"] == "S01/S1, S02/S1"
    assert out.loc[0, "hash"] == "aaaaaaaaaaaa"


def test_a_database_made_by_version_0_1_is_upgraded_in_place(tmp_path):
    path = tmp_path / "old.sqlite"
    with closing(sqlite3.connect(path)) as old:
        old.executescript(
            """
            CREATE TABLE subjects (subject_id TEXT PRIMARY KEY);
            CREATE TABLE recordings (
                recording_id INTEGER PRIMARY KEY,
                subject_id TEXT NOT NULL REFERENCES subjects (subject_id),
                session TEXT NOT NULL DEFAULT 'S1', fs_hz REAL, preprocess TEXT,
                UNIQUE (subject_id, session));
            INSERT INTO subjects VALUES ('S01');
            INSERT INTO recordings (subject_id, session, fs_hz) VALUES ('S01', 'S1', 1000.0);
            """
        )
    with closing(connect(path)) as conn:
        columns = [r[1] for r in conn.execute("PRAGMA table_info(recordings)")]
        assert "content_sha256" in columns
        assert conn.execute("SELECT fs_hz, content_sha256 FROM recordings").fetchall() == [
            (1000.0, None)
        ]
        register_recording(conn, "S01", "S1", content_sha256="cc" * 32)
    with closing(connect(path)) as again:  # opening it twice is harmless
        assert again.execute("SELECT content_sha256 FROM recordings").fetchone() == ("cc" * 32,)


def test_the_example_runs_on_an_empty_database():
    with closing(connect()) as conn:
        assert query(conn, read_example("duplicate_recordings")).empty
