"""End-to-end demo on a tiny cohort, plus the README quickstart."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

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

    assert json.loads((out / "summary.json").read_text())["n_trials"] == N_TRIALS
    report = (out / "report.md").read_text()
    assert "synthetic" in report and "| lda | loso | within_group |" in report


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
