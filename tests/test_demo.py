"""End-to-end demo on a tiny cohort, plus the README quickstart."""

from __future__ import annotations

import json
import re
import runpy
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest

from biosigkit import connect
from biosigkit.demo import run_demo
from biosigkit.storage import list_examples

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
N_SUBJECTS, TRIALS_PER_LEVEL = 3, 3
N_TRIALS = N_SUBJECTS * 4 * TRIALS_PER_LEVEL


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    messages: list[str] = []
    summary = run_demo(
        out, seed=3, n_subjects=N_SUBJECTS, trials_per_level=TRIALS_PER_LEVEL, log=messages.append
    )
    return out, summary, messages


def test_demo_writes_every_artifact(demo):
    out, _, messages = demo
    for name in (
        "report.md",
        "summary.json",
        "cv_results.csv",
        "loso_folds_lda_zscore.csv",
        "eeg_alpha_erd.csv",
        "force_tracking_trials.csv",
        "qc.csv",
        "trial_checks.csv",
        "dashboard.html",
        "biosigkit_demo.sqlite",
    ):
        assert (out / name).stat().st_size > 0, name
    for figure in ("emg_overview", "cv_scenarios", "confusion_matrix", "eeg_erd"):
        assert (out / "figures" / f"{figure}.png").read_bytes().startswith(PNG_MAGIC), figure
    assert sorted(p.stem for p in (out / "sql").glob("*.csv")) == sorted(list_examples())
    assert [m for m in messages if m.startswith("[")][-1].startswith("[6/6]")


def test_demo_summary_is_internally_consistent(demo):
    out, summary, _ = demo
    assert summary["n_subjects"] == N_SUBJECTS
    assert summary["n_trials"] == N_TRIALS
    assert summary["n_classes"] == 4 and summary["chance"] == 0.25
    assert summary["episodes_from_sql"] == summary["episodes_expected"] == N_TRIALS
    assert all(hz == [60.0, 120.0] for hz in summary["mains_detected"].values())

    detection = summary["detection"]
    assert detection["detected"] >= 0.9 * detection["trials"]
    assert detection["false_positives"] <= 2
    assert detection["median_onset_latency_s"] < 0.5

    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["n_trials"] == N_TRIALS
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "synthetic" in report and "| lda | loso | within_group |" in report


def test_demo_data_checks_find_nothing_wrong_in_clean_simulated_data(demo):
    out, summary, _ = demo
    checks = summary["checks"]
    assert checks["trials_checked"] == checks["trials_valid"] == N_TRIALS
    assert checks["plateau_found"] == checks["plateau_trials"] == N_TRIALS
    assert checks["duplicate_groups"] == 0
    assert checks["channels_checked"] == 2 * N_SUBJECTS
    assert checks["max_saturated_fraction"] == 0.0
    assert 4.0 < checks["median_hold_s"] <= 6.0  # 6 s trials with 0.5 s ramps
    assert checks["median_hold_s"] > checks["median_fixed_trim_window_s"]  # the fixed 1 s trim
    qc = pd.read_csv(out / "qc.csv")
    assert qc["subject"].nunique() == N_SUBJECTS and {"mains_hz", "ok", "issues"} <= set(qc.columns)
    assert (pd.read_csv(out / "trial_checks.csv")["valid"]).all()
    assert "## Data checks" in (out / "report.md").read_text(encoding="utf-8")


def test_demo_text_files_are_utf8_with_unix_newlines(demo):
    out, _, _ = demo
    for name in ("report.md", "summary.json", "dashboard.html"):
        raw = (out / name).read_bytes()
        raw.decode("utf-8")  # raises if the file was written in the platform's default encoding
        assert b"\r" not in raw, name


def test_demo_dashboard_shows_the_numbers_of_the_demo_run(demo):
    out, summary, _ = demo
    page = (out / "dashboard.html").read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S)
    assert match
    data = json.loads(match.group(1))
    assert data["n_rows"] == N_TRIALS and len(data["subjects"]) == N_SUBJECTS
    assert data["labels"] == ["10%", "30%", "50%", "70%"] and data["label_scale"] == "ordinal"
    # the validation chart reuses the demo's cross-validation instead of running it again
    shown = {(r["model"], r["scenario"]): r["accuracy"] for r in data["validation"]["results"]}
    for row in summary["scenarios"]:
        key = (row["model"], f"{row['scheme']}|{row['normalization']}")
        assert shown[key] == pytest.approx(row["accuracy"], abs=1e-5)
    assert "dashboard.html" in (out / "report.md").read_text(encoding="utf-8")


def test_demo_hashes_every_recording_and_finds_no_duplicates(demo):
    out, _, _ = demo
    with closing(connect(out / "biosigkit_demo.sqlite")) as conn:
        digests = [r[0] for r in conn.execute("SELECT content_sha256 FROM recordings")]
    assert len(digests) == N_SUBJECTS and len(set(digests)) == N_SUBJECTS
    assert all(d and len(d) == 64 for d in digests)
    assert (out / "sql" / "duplicate_recordings.csv").read_text(
        encoding="utf-8"
    ).strip() == "hash,n_recordings,recordings"


def test_demo_scenarios_cover_the_three_validation_schemes(demo):
    _, summary, _ = demo
    combos = {(r["scheme"], r["normalization"]) for r in summary["scenarios"]}
    assert combos == {("kfold", "none"), ("loso", "none"), ("loso", "within_group")}
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in summary["scenarios"])
    best = max(r["accuracy"] for r in summary["scenarios"] if r["normalization"] == "within_group")
    assert best > 0.5  # chance is 0.25


def test_demo_is_reproducible(demo, tmp_path):
    _, summary, _ = demo
    again = run_demo(
        tmp_path, seed=3, n_subjects=N_SUBJECTS, trials_per_level=TRIALS_PER_LEVEL, log=None
    )
    assert again["scenarios"] == summary["scenarios"]
    assert again["eeg"] == summary["eeg"]


def test_readme_quickstart_runs(capsys):
    script = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"
    if not script.exists():  # e.g. tests run against an installed wheel
        pytest.skip("examples/ is not available")
    runpy.run_path(str(script), run_name="__main__")
    output = capsys.readouterr().out
    assert "notch applied at [60.0, 120.0]" in output
    assert "within_group" in output


def test_data_checks_example_runs(capsys):
    script = Path(__file__).resolve().parents[1] / "examples" / "data_checks.py"
    if not script.exists():  # e.g. tests run against an installed wheel
        pytest.skip("examples/ is not available")
    runpy.run_path(str(script), run_name="__main__")
    output = capsys.readouterr().out
    # the damage the script inflicts is found, and said in words
    assert "of samples clipped" in output
    assert "of samples are NaN or infinite" in output
    assert "nonfinite;saturated" in output
    # the usable trials still get a hold phase, and the copy is found by its samples
    assert "usable trials; hold found in" in output
    assert "S01/S1, S09/S1" in output
