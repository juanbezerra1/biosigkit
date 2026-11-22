"""SQLite persistence for feature tables, plus bundled analytical SQL examples.

Only the standard-library ``sqlite3`` module is used, with parameterised statements throughout.
The schema (``sql/schema.sql``) is normalised: subjects -> recordings -> segments -> features,
where ``segments`` are either sliding windows or trials and ``features`` is a long table.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd

from .tables import features_to_long, features_to_wide, infer_feature_columns


def _sql_dir() -> Any:
    return resources.files("biosigkit") / "sql"


def init_db(conn: sqlite3.Connection) -> None:
    """Create the tables and indexes if they do not exist yet."""
    conn.executescript((_sql_dir() / "schema.sql").read_text(encoding="utf-8"))
    conn.commit()


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open (creating if needed) a database with foreign-key enforcement switched on."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    init_db(conn)
    return conn


def register_recording(
    conn: sqlite3.Connection,
    subject: str,
    session: str = "S1",
    *,
    fs: float | None = None,
    preprocess: Mapping[str, Any] | None = None,
) -> int:
    """Insert the subject/recording if new, optionally store ``fs`` and preprocessing provenance.

    Returns the ``recording_id``. Existing values are only overwritten by non-``None`` arguments.
    """
    with conn:
        conn.execute("INSERT OR IGNORE INTO subjects (subject_id) VALUES (?)", (subject,))
        conn.execute(
            "INSERT OR IGNORE INTO recordings (subject_id, session) VALUES (?, ?)",
            (subject, session),
        )
        if fs is not None or preprocess is not None:
            conn.execute(
                "UPDATE recordings SET fs_hz = COALESCE(?, fs_hz), "
                "preprocess = COALESCE(?, preprocess) WHERE subject_id = ? AND session = ?",
                (
                    fs,
                    None if preprocess is None else json.dumps(preprocess, default=str),
                    subject,
                    session,
                ),
            )
        row = conn.execute(
            "SELECT recording_id FROM recordings WHERE subject_id = ? AND session = ?",
            (subject, session),
        ).fetchone()
    return int(row[0])


def save_feature_table(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    *,
    kind: str,
    subject_col: str = "subject",
    session_col: str = "session",
    feature_cols: Sequence[str] | None = None,
) -> int:
    """Store a table produced by ``window_feature_table`` / ``trial_feature_table``.

    ``kind`` is ``"window"`` or ``"trial"``. Rows need ``t_start``, ``t_stop`` and the subject
    column; ``label`` and the session column are optional (session defaults to ``"S1"``).
    Each (subject, session) is written in one transaction. Returns the number of segments saved.
    """
    if kind not in {"window", "trial"}:
        raise ValueError("kind must be 'window' or 'trial'")
    missing = [c for c in (subject_col, "t_start", "t_stop") if c not in df.columns]
    if missing:
        raise ValueError(f"table is missing columns: {missing}")
    cols = list(feature_cols) if feature_cols is not None else infer_feature_columns(df)
    if not cols:
        raise ValueError("no feature columns found")
    work = df.copy()
    if session_col not in work.columns:
        work[session_col] = "S1"
    if "label" not in work.columns:
        work["label"] = None

    total = 0
    for (subject, session), group in work.groupby([subject_col, session_col], sort=False):
        recording_id = register_recording(conn, str(subject), str(session))
        with conn:
            first = conn.execute("SELECT COALESCE(MAX(segment_id), 0) FROM segments").fetchone()[0]
            ids = list(range(first + 1, first + 1 + len(group)))
            labels = [None if pd.isna(v) else str(v) for v in group["label"]]
            conn.executemany(
                "INSERT INTO segments (segment_id, recording_id, kind, t_start, t_stop, label) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                zip(
                    ids,
                    [recording_id] * len(ids),
                    [kind] * len(ids),
                    group["t_start"].astype(float).tolist(),
                    group["t_stop"].astype(float).tolist(),
                    labels,
                    strict=True,
                ),
            )
            long = features_to_long(group.assign(segment_id=ids), ["segment_id"], cols)
            conn.executemany(
                "INSERT INTO features (segment_id, channel, feature, value) VALUES (?, ?, ?, ?)",
                zip(
                    long["segment_id"].tolist(),
                    long["channel"].tolist(),
                    long["feature"].tolist(),
                    long["value"].astype(float).tolist(),  # NaN is stored as NULL
                    strict=True,
                ),
            )
        total += len(group)
    return total


_LOAD_SQL = """
SELECT s.segment_id, r.subject_id AS subject, r.session, s.t_start, s.t_stop, s.label,
       f.channel, f.feature, f.value
FROM features   AS f
JOIN segments   AS s ON s.segment_id   = f.segment_id
JOIN recordings AS r ON r.recording_id = s.recording_id
WHERE s.kind = ?
ORDER BY s.segment_id, f.rowid
"""


def load_feature_table(conn: sqlite3.Connection, kind: str = "trial") -> pd.DataFrame:
    """Read segments of one ``kind`` back into the wide ``<channel>_<feature>`` layout."""
    long = pd.read_sql_query(_LOAD_SQL, conn, params=(kind,))
    if long.empty:
        return pd.DataFrame()
    meta_cols = ["segment_id", "subject", "session", "t_start", "t_stop", "label"]
    meta = long.drop_duplicates("segment_id")[meta_cols]
    wide = features_to_wide(long, "segment_id")
    return meta.merge(wide, left_on="segment_id", right_index=True).reset_index(drop=True)


def query(
    conn: sqlite3.Connection, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()
) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame. Pass values via ``params``, never by formatting."""
    return pd.read_sql_query(sql, conn, params=params)


def _example_files() -> dict[str, Any]:
    files = sorted(
        (p for p in (_sql_dir() / "queries").iterdir() if p.name.endswith(".sql")),
        key=lambda p: p.name,
    )
    return {re.sub(r"^\d+_", "", p.name.removesuffix(".sql")): p for p in files}


def list_examples() -> list[str]:
    """Names of the bundled example queries (numeric prefix and extension stripped)."""
    return list(_example_files())


def read_example(name: str) -> str:
    """SQL text of a bundled example, e.g. ``read_example("activation_episodes")``."""
    files = _example_files()
    key = re.sub(r"^\d+_", "", name.removesuffix(".sql"))
    if key not in files:
        raise KeyError(f"unknown example {name!r}; available: {list(files)}")
    return files[key].read_text(encoding="utf-8")
