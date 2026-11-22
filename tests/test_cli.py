"""Command-line interface: happy paths and clean, traceback-free failures."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from contextlib import closing

import pandas as pd
import pytest
from typer.testing import CliRunner

from biosigkit import (
    __version__,
    connect,
    make_cohort,
    make_session,
    trial_feature_table,
    write_csv,
)
from biosigkit.cli import app
from biosigkit.storage import save_feature_table

runner = CliRunner()


def run(*args: object):
    result = runner.invoke(app, [str(a) for a in args])
    assert "Traceback" not in result.output, result.output
    return result


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """A signal CSV (with a time column), the same without it, and an events CSV."""
    folder = tmp_path_factory.mktemp("recording")
    session = make_session("S01", trials_per_level=1, seed=5)
    write_csv(session.emg, folder / "emg_time.csv")
    session.emg.to_frame().drop(columns="time").to_csv(folder / "emg_plain.csv", index=False)
    session.events[["onset", "offset", "label"]].to_csv(folder / "events.csv", index=False)
    return folder, session


@pytest.fixture(scope="module")
def cohort_csv(tmp_path_factory):
    """Trial-level feature table of four simulated subjects, saved as CSV."""
    tables = [
        trial_feature_table(
            s.emg, s.events, trim_s=1.0, meta={"subject": s.subject, "session": "S1"}
        )
        for s in make_cohort(4, trials_per_level=3, seed=2)
    ]
    path = tmp_path_factory.mktemp("cohort") / "trials.csv"
    pd.concat(tables, ignore_index=True).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def database(tmp_path_factory, cohort_csv):
    path = tmp_path_factory.mktemp("db") / "demo.sqlite"
    with closing(connect(path)) as conn:
        save_feature_table(conn, pd.read_csv(cohort_csv), kind="trial")
    return path


# ------------------------------------------------------------------------------- basics
def test_version_flag():
    result = run("--version")
    assert result.exit_code == 0
    assert result.output.strip() == f"biosigkit {__version__}"


def test_sql_list_and_show():
    listed = run("sql", "list")
    assert listed.exit_code == 0
    assert "activation_episodes" in listed.output.split()
    shown = run("sql", "show", "activation_episodes")
    assert shown.exit_code == 0 and "ROW_NUMBER" in shown.output


def test_sql_show_reports_unknown_examples():
    result = run("sql", "show", "nope")
    assert result.exit_code == 1 and "unknown example" in result.output


# ------------------------------------------------------------------------------ features
def test_features_in_window_mode_with_events(tmp_path, recording):
    folder, session = recording
    out = tmp_path / "windows.csv"
    result = run(
        "features", folder / "emg_time.csv", "--time-col", "time", "--events",
        folder / "events.csv", "--output", out, "--subject", "S07",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Wrote" in result.output
    table = pd.read_csv(out)
    assert {"subject", "session", "t_start", "t_stop", "label", "emg1_rms", "emg2_rms"} <= set(
        table.columns
    )
    assert set(table["subject"]) == {"S07"}
    assert set(table["label"]) == {"rest", *session.events["label"]}


def test_features_in_trial_mode_without_a_time_column(tmp_path, recording):
    folder, session = recording
    out = tmp_path / "trials.csv"
    result = run(
        "features", folder / "emg_plain.csv", "--fs", 1000, "--events", folder / "events.csv",
        "--mode", "trial", "--trim", 1.0, "--output", out, "--no-preprocess",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    table = pd.read_csv(out)
    assert len(table) == len(session.events)
    assert table["t_start"].round(3).tolist() == (session.events["onset"] + 1.0).round(3).tolist()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--mode", "trial"], "needs --events"),
        (["--mode", "weekly"], "--mode must be"),
        (["--kind", "ecg"], "--kind must be"),
        (["--fs", "1000", "--time-col", "nope"], "nope"),
    ],
)
def test_features_fails_cleanly(recording, tmp_path, extra, message):
    folder, _ = recording
    args = ["features", folder / "emg_time.csv", "--time-col", "time", "-o", tmp_path / "x.csv"]
    if "--time-col" in extra:
        args = ["features", folder / "emg_time.csv", "-o", tmp_path / "x.csv"]
    result = run(*args, *extra)
    assert result.exit_code == 1
    assert message in result.output


# ------------------------------------------------------------------------------ evaluate
def test_evaluate_prints_metrics_and_a_confusion_matrix(cohort_csv):
    result = run("evaluate", cohort_csv, "--model", "lda", "--normalize")
    assert result.exit_code == 0, result.output
    assert "normalize=True" in result.output
    assert "accuracy" in result.output and "Confusion matrix" in result.output


def test_evaluate_supports_other_schemes(cohort_csv):
    result = run("evaluate", cohort_csv, "--model", "svm_linear", "--scheme", "group_kfold")
    assert result.exit_code == 0, result.output
    assert "scheme=group_kfold" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--model", "transformer"], "unknown model"),
        (["--scheme", "bogus"], "scheme must be"),
        (["--group", "nobody"], "nobody"),
    ],
)
def test_evaluate_fails_cleanly(cohort_csv, args, message):
    result = run("evaluate", cohort_csv, *args)
    assert result.exit_code == 1
    assert message in result.output


# ---------------------------------------------------------------------------------- sql run
def test_sql_run_executes_bundled_and_custom_queries(database):
    bundled = run("sql", "run", database, "--example", "data_quality")
    assert bundled.exit_code == 0 and "balance_flag" in bundled.output
    custom = run("sql", "run", database, "-q", "SELECT COUNT(*) AS n_subjects FROM subjects")
    assert custom.exit_code == 0 and "n_subjects" in custom.output
    assert custom.output.split()[-1] == "4"


def test_sql_run_needs_exactly_one_source(database):
    neither = run("sql", "run", database)
    both = run("sql", "run", database, "-e", "data_quality", "-q", "SELECT 1")
    assert neither.exit_code == both.exit_code == 1
    assert "exactly one" in neither.output


def test_sql_run_reports_sql_errors_without_a_traceback(database):
    result = run("sql", "run", database, "-q", "SELEC nothing")
    assert result.exit_code == 1 and "syntax error" in result.output


def test_sql_run_is_read_only(database):
    result = run("sql", "run", database, "-q", "DROP TABLE subjects")
    assert result.exit_code == 1
    with closing(sqlite3.connect(database)) as conn:  # the table is still there
        assert conn.execute("SELECT COUNT(*) FROM subjects").fetchone()[0] == 4


# ------------------------------------------------------------------------------------ demo
def test_demo_rejects_a_single_subject(tmp_path):
    result = run("demo", "--out", tmp_path / "out", "--subjects", 1)
    assert result.exit_code == 1 and "at least 2" in result.output


def test_module_entry_point_matches_the_console_script():
    result = subprocess.run(
        [sys.executable, "-m", "biosigkit", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"biosigkit {__version__}"
