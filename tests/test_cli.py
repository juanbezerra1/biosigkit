"""Command-line interface: happy paths and clean, traceback-free failures."""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
from contextlib import closing

import numpy as np
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


# ------------------------------------------------------------------------ export, dashboard
def embedded_data(path):
    """The JSON a dashboard page carries."""
    page = path.read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S)
    assert match, "the page has no data element"
    return json.loads(match.group(1))


def test_export_writes_the_feature_table_back_to_a_csv(database, cohort_csv, tmp_path):
    out = tmp_path / "features.csv"
    result = run("export", database, "-o", out)
    assert result.exit_code == 0, result.output
    assert "Wrote 48 rows x 16 columns" in result.output
    exported = pd.read_csv(out)
    original = pd.read_csv(cohort_csv)
    assert {"segment_id", "subject", "label", "emg1_mav", "emg2_ssc"} <= set(exported.columns)
    assert len(exported) == len(original)
    # the numbers survive the round trip through SQL
    merged = exported.sort_values(["subject", "t_start"]).reset_index(drop=True)
    ref = original.sort_values(["subject", "t_start"]).reset_index(drop=True)
    np.testing.assert_allclose(merged["emg1_rms"], ref["emg1_rms"])
    assert merged["label"].tolist() == ref["label"].tolist()
    # and the file is ready for the other commands
    assert run("evaluate", out, "--model", "lda", "--normalize").exit_code == 0


def test_export_does_not_change_the_database(database, tmp_path):
    before = database.read_bytes()
    run("export", database, "-o", tmp_path / "features.csv")
    assert database.read_bytes() == before


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--kind", "window"], "holds no window segments (try --kind trial)"),
        (["--kind", "epoch"], "--kind must be"),
    ],
)
def test_export_fails_cleanly(database, tmp_path, args, message):
    result = run("export", database, "-o", tmp_path / "x.csv", *args)
    assert result.exit_code == 1 and message in result.output
    assert not (tmp_path / "x.csv").exists()


def test_export_refuses_a_file_that_is_not_a_biosigkit_database(tmp_path, cohort_csv):
    (tmp_path / "text.db").write_text("not a database", encoding="utf-8")
    result = run("export", tmp_path / "text.db", "-o", tmp_path / "x.csv")
    assert result.exit_code == 1 and "not a biosigkit database" in result.output
    other = tmp_path / "other.sqlite"  # a real SQLite file, but not ours
    with closing(sqlite3.connect(other)) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER)")
    result = run("export", other, "-o", tmp_path / "x.csv")
    assert result.exit_code == 1 and "not a biosigkit database" in result.output
    assert "notes" in result.output


def test_dashboard_from_a_csv(cohort_csv, tmp_path):
    out = tmp_path / "page.html"
    result = run("dashboard", cohort_csv, "-o", out, "--title", "Four subjects")
    assert result.exit_code == 0, result.output
    assert "48 rows, 4 subjects, 4 labels, 10 features" in result.output
    page = out.read_text(encoding="utf-8")
    assert "<title>Four subjects</title>" in page
    data = embedded_data(out)
    assert data["source"] == "trials.csv" and data["id_name"] == "trial"
    assert len(data["validation"]["results"]) == 12  # 4 models x 3 validation schemes
    assert b"\r" not in out.read_bytes()


def test_the_output_option_is_spelled_both_ways(cohort_csv, database, tmp_path):
    # `demo` says --out, `features` says --output; the new commands accept both
    for name in ("--out", "--output", "-o"):
        page = tmp_path / f"page{name}.html"
        assert run("dashboard", cohort_csv, name, page, "--no-validation").exit_code == 0
        assert page.stat().st_size > 0
        table = tmp_path / f"table{name}.csv"
        assert run("export", database, name, table).exit_code == 0
        assert table.stat().st_size > 0


def test_dashboard_from_a_database_reads_the_stored_table(database, tmp_path):
    out = tmp_path / "page.html"
    result = run("dashboard", database, "-o", out, "--no-validation")
    assert result.exit_code == 0, result.output
    data = embedded_data(out)
    assert data["n_rows"] == 48 and data["id_name"] == "segment"
    assert data["validation"] is None
    assert run("dashboard", database, "-o", out, "--kind", "window").exit_code == 1


def test_dashboard_options_reach_the_page(cohort_csv, tmp_path):
    out = tmp_path / "page.html"
    result = run(
        "dashboard", cohort_csv, "-o", out,
        "--labels", "10%, 70%",
        "--features", "emg1_mav,emg2_rms",
        "--models", "lda,svm_linear",
        "--label-scale", "nominal",
        "--max-rows", 20,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "a random sample of 24 rows (--max-rows 20)" in result.output
    data = embedded_data(out)
    assert data["labels"] == ["10%", "70%"] and data["label_scale"] == "nominal"
    assert sorted(data["values"]) == ["emg1_mav", "emg2_rms"] and data["n_rows"] == 20
    assert [m["key"] for m in data["validation"]["models"]] == ["lda", "svm_linear"]


def test_dashboard_says_why_it_left_the_validation_chart_out(cohort_csv, tmp_path):
    two = tmp_path / "two.csv"
    table = pd.read_csv(cohort_csv)
    table[table["subject"].isin(["S01", "S02"])].to_csv(two, index=False)
    result = run("dashboard", two, "-o", tmp_path / "page.html")
    assert result.exit_code == 0, result.output
    assert "no validation chart: validation needs at least 3 subjects" in result.output
    assert embedded_data(tmp_path / "page.html")["validation"]["error"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--label-col", "nothing"], "column 'nothing' not found"),
        (["--subject-col", "nobody"], "column 'nobody' not found"),
        (["--features", "emg1_mav,nope"], "feature column(s) not found"),
        (["--features", "session"], "not numeric"),
        (["--models", "transformer"], "unknown model"),
        (["--models", ""], "models is empty"),
        (["--labels", "99%"], "none of the labels"),
        (["--label-scale", "rainbow"], "--label-scale must be"),
        (["--kind", "epoch"], "--kind must be"),
        (["--max-rows", "0"], "--max-rows must be at least 1"),
    ],
)
def test_dashboard_fails_cleanly(cohort_csv, tmp_path, args, message):
    out = tmp_path / "page.html"
    result = run("dashboard", cohort_csv, "-o", out, *args)
    assert result.exit_code == 1 and message in result.output
    assert not out.exists()


def test_dashboard_refuses_more_labels_than_it_can_colour(tmp_path):
    rows = [
        {"subject": f"P{s}", "label": f"c{i}", "f1": float(i)} for s in range(3) for i in range(9)
    ]
    path = tmp_path / "many.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = run("dashboard", path, "-o", tmp_path / "page.html", "--no-validation")
    assert result.exit_code == 1 and "9 labels" in result.output and "--labels" in result.output


def test_dashboard_reports_an_unreadable_csv_without_a_traceback(tmp_path):
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    result = run("dashboard", tmp_path / "empty.csv", "-o", tmp_path / "page.html")
    assert result.exit_code == 1 and "Error:" in result.output


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


# ---------------------------------------------------------------- trial checks in `features`
@pytest.fixture
def late_events(tmp_path, recording):
    """The recording's events plus one trial that starts after the recording ends."""
    _, session = recording
    events = session.events[["onset", "offset", "label"]].copy()
    end = session.emg.duration
    events.loc[len(events)] = [end + 1.0, end + 7.0, "late"]
    path = tmp_path / "events_late.csv"
    events.to_csv(path, index=False)
    return path


def trial_args(folder, events, out, *extra):
    return [
        "features", folder / "emg_plain.csv", "--fs", 1000, "--events", events, "--mode", "trial",
        "--no-preprocess", "--output", out, *extra,
    ]  # fmt: skip


def test_trial_mode_names_the_bad_trial_and_can_skip_it(tmp_path, recording, late_events):
    folder, session = recording
    out = tmp_path / "t.csv"
    failed = run(*trial_args(folder, late_events, out))
    assert failed.exit_code == 1
    assert "1 of 5 trial(s) rejected" in failed.output and "outside the recording" in failed.output
    assert "--on-invalid skip" in failed.output
    assert not out.exists()

    rejected_csv = tmp_path / "rejected.csv"
    ok = run(*trial_args(folder, late_events, out, "--on-invalid", "skip",
                         "--rejected-out", rejected_csv))  # fmt: skip
    assert ok.exit_code == 0, ok.output
    assert "Skipped 1 trial(s): 1 outside_recording" in ok.output
    rejected = pd.read_csv(rejected_csv)
    assert rejected["label"].tolist() == ["late"]
    assert rejected["reason"].tolist() == ["outside_recording"]
    assert "valid" not in rejected.columns
    assert len(pd.read_csv(out)) == len(session.events)


def test_check_quality_rejects_trials_with_missing_samples(tmp_path, recording):
    folder, session = recording
    signal = pd.read_csv(folder / "emg_plain.csv")
    first = session.events.iloc[0]
    signal.loc[round((first.onset + 2.0) * 1000), "emg1"] = np.nan
    holes = tmp_path / "holes.csv"
    signal.to_csv(holes, index=False)
    args = ["features", holes, "--fs", 1000, "--events", folder / "events.csv", "--mode", "trial",
            "--no-preprocess", "--output", tmp_path / "t.csv"]  # fmt: skip

    failed = run(*args, "--check-quality")
    assert failed.exit_code == 1
    assert "emg1" in failed.output and "NaN or infinite" in failed.output

    skipped = run(*args, "--check-quality", "--on-invalid", "skip")
    assert skipped.exit_code == 0 and "Skipped 1 trial(s): 1 nonfinite" in skipped.output
    assert len(pd.read_csv(tmp_path / "t.csv")) == len(session.events) - 1


def test_trial_options_are_refused_in_window_mode(recording, tmp_path):
    folder, _ = recording
    result = run("features", folder / "emg_time.csv", "--time-col", "time", "--check-quality",
                 "-o", tmp_path / "x.csv")  # fmt: skip
    assert result.exit_code == 1 and "only apply to --mode trial" in result.output
    result = run("features", folder / "emg_time.csv", "--time-col", "time", "--on-invalid", "maybe",
                 "-o", tmp_path / "x.csv")  # fmt: skip
    assert result.exit_code == 1 and "--on-invalid must be" in result.output


# --------------------------------------------------------------------------- qc, duplicates
@pytest.fixture(scope="module")
def csv_files(tmp_path_factory):
    folder = tmp_path_factory.mktemp("qc")
    rng = np.random.default_rng(21)
    clean = pd.DataFrame(rng.standard_normal((10000, 2)), columns=["emg1", "emg2"])
    clipped = clean.copy()
    clipped["emg2"] = np.clip(3.0 * clipped["emg2"], -2.0, 2.0)
    clean.to_csv(folder / "clean.csv", index=False)
    clean.to_csv(folder / "copy_of_clean.csv", index=False)
    clipped.to_csv(folder / "clipped.csv", index=False)
    other = pd.DataFrame(rng.standard_normal((10000, 2)), columns=["emg1", "emg2"])
    other.to_csv(folder / "other.csv", index=False)
    return folder


def test_qc_reports_each_channel_and_strict_sets_the_exit_status(csv_files, tmp_path):
    clean = run("qc", csv_files / "clean.csv", "--fs", 1000, "--strict")
    assert clean.exit_code == 0 and "2 of 2 channel(s) ok." in clean.output

    out = tmp_path / "qc.csv"
    clipped = run("qc", csv_files / "clipped.csv", "--fs", 1000, "-o", out)
    assert clipped.exit_code == 0  # a report, not a gate, unless --strict
    assert "emg2: " in clipped.output and "clipped" in clipped.output
    assert "1 of 2 channel(s) ok." in clipped.output
    assert pd.read_csv(out).set_index("channel")["ok"].to_dict() == {"emg1": True, "emg2": False}

    strict = run("qc", csv_files / "clipped.csv", "--fs", 1000, "--strict")
    assert strict.exit_code == 1


def test_qc_needs_a_sampling_rate(csv_files):
    result = run("qc", csv_files / "clean.csv")
    assert result.exit_code == 1 and "provide fs or time_col" in result.output


def test_duplicates_finds_identical_recordings(csv_files):
    files = [csv_files / n for n in ("clean.csv", "clipped.csv", "copy_of_clean.csv")]
    result = run("duplicates", *files, "--fs", 1000)
    assert result.exit_code == 0, result.output
    assert "recording" in result.output
    assert "clean.csv" in result.output and "copy_of_clean.csv" in result.output
    assert run("duplicates", *files, "--fs", 1000, "--strict").exit_code == 1


def test_duplicates_also_reports_a_channel_shared_by_two_files(csv_files):
    files = [csv_files / "clean.csv", csv_files / "clipped.csv"]  # same emg1, different emg2
    result = run("duplicates", *files, "--fs", 1000)
    assert result.exit_code == 0 and "channel" in result.output and "/emg1" in result.output


def test_duplicates_says_so_when_there_are_none(csv_files):
    files = [csv_files / "clean.csv", csv_files / "other.csv"]
    result = run("duplicates", *files, "--fs", 1000, "--strict")
    assert result.exit_code == 0 and "No duplicates among 2 recording(s)." in result.output
