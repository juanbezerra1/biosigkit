"""The dashboard: the data behind the page, and the page itself."""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biosigkit import (
    dashboard_data,
    make_cohort,
    render_dashboard,
    trial_feature_table,
    within_group_zscore,
    write_dashboard,
)
from biosigkit.ml import compare_scenarios


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    """Trial features of four simulated subjects (48 trials, labels 10 %, 30 %, 50 %, 70 %)."""
    tables = [
        trial_feature_table(
            s.emg, s.events, trim_s=1.0, meta={"subject": s.subject, "session": "S1"}
        )
        for s in make_cohort(4, trials_per_level=3, seed=2)
    ]
    return pd.concat(tables, ignore_index=True)


def small_table(labels, n_subjects=3, reps=3, seed=0) -> pd.DataFrame:
    """One feature, ``n_subjects`` subjects, every label ``reps`` times in each subject."""
    rng = np.random.default_rng(seed)
    rows = [
        {"subject": f"P{s + 1}", "label": lab, "f1": float(i + s + rng.normal(0, 0.1))}
        for s in range(n_subjects)
        for i, lab in enumerate(labels)
        for _ in range(reps)
    ]
    return pd.DataFrame(rows)


def eta_squared_reference(values: pd.Series, labels: pd.Series) -> float:
    """One-way ANOVA effect size written out the long way, as an independent check."""
    grand = values.mean()
    between = sum(len(g) * (g.mean() - grand) ** 2 for _, g in values.groupby(labels))
    return float(between / ((values - grand) ** 2).sum())


def embedded_data(page: str) -> dict:
    """The JSON the page carries, parsed the way a browser would read the script element."""
    match = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S)
    assert match, "the data element is missing"
    return json.loads(match.group(1))


# ------------------------------------------------------------------------------ the data
def test_data_describes_the_table(cohort):
    d = dashboard_data(cohort, validate=False)
    assert d["n_rows"] == len(cohort) == 48
    assert d["subjects"] == ["S01", "S02", "S03", "S04"]
    assert d["labels"] == ["10%", "30%", "50%", "70%"] and d["label_scale"] == "ordinal"
    assert d["id_name"] == "trial" and d["sampled_from"] is None and d["dropped_rows"] == 0
    assert d["validation"] is None

    names = [f["name"] for f in d["features"]]
    assert names and all(n.startswith("emg") for n in names)  # no subject, trial, times, level
    assert set(d["values"]) == set(names)
    assert all(len(v) == 48 for v in d["values"].values())
    assert {len(v) for v in d["rows"].values()} == {48}
    # rows are coded as indices into subjects and labels
    first = cohort.iloc[0]
    assert d["subjects"][d["rows"]["subject"][0]] == first["subject"]
    assert d["labels"][d["rows"]["label"][0]] == first["label"]


def test_features_are_ranked_by_eta_squared_after_the_subject_zscore(cohort):
    d = dashboard_data(cohort, validate=False)
    etas = [f["eta_z"] for f in d["features"]]
    assert etas == sorted(etas, reverse=True)

    names = [f["name"] for f in d["features"]]
    z = within_group_zscore(cohort, names, "subject")
    for f in d["features"][:3] + d["features"][-2:]:
        assert f["eta_raw"] == pytest.approx(
            eta_squared_reference(cohort[f["name"]], cohort["label"]), rel=1e-4
        )
        assert f["eta_z"] == pytest.approx(
            eta_squared_reference(z[f["name"]], cohort["label"]), rel=1e-4
        )
    # the simulated hold force scales the EMG per subject, so the z-score is what shows it
    assert d["features"][0]["eta_z"] > d["features"][0]["eta_raw"]


def test_features_option_picks_and_orders_the_columns(cohort):
    d = dashboard_data(cohort, features=["emg1_rms", "emg1_mav"], validate=False)
    assert {f["name"] for f in d["features"]} == {"emg1_rms", "emg1_mav"}
    assert list(d["values"]) == ["emg1_rms", "emg1_mav"]  # the columns keep the order asked for


@pytest.mark.parametrize(
    ("labels", "scale", "ordered", "expected_scale"),
    [
        (["70%", "10%", "50%", "30%"], "auto", ["10%", "30%", "50%", "70%"], "ordinal"),
        (["20 N", "5 N", "10 N"], "auto", ["5 N", "10 N", "20 N"], "ordinal"),
        (["low", "mid", "high"], "auto", ["low", "mid", "high"], "nominal"),  # file order kept
        (
            ["1", "2", "3", "4", "5"],
            "auto",
            ["1", "2", "3", "4", "5"],
            "nominal",
        ),  # ramp: 3-4 steps
        (["1", "2"], "auto", ["1", "2"], "nominal"),
        (["10%", "30%", "50%"], "nominal", ["10%", "30%", "50%"], "nominal"),
        (["b3", "b10", "b2"], "ordinal", ["b2", "b3", "b10"], "ordinal"),  # natural order
    ],
)
def test_label_order_and_colour_scale(labels, scale, ordered, expected_scale):
    d = dashboard_data(small_table(labels), label_scale=scale, validate=False)
    assert d["labels"] == ordered and d["label_scale"] == expected_scale


def test_subjects_are_in_natural_order():
    table = small_table(["a", "b"], n_subjects=11)
    subjects = dashboard_data(table, validate=False)["subjects"]
    assert subjects[:3] == ["P1", "P2", "P3"] and subjects[-1] == "P11"


def test_missing_and_infinite_values_become_null_and_the_json_stays_valid(cohort):
    table = cohort.copy()
    table.loc[3, "emg1_mav"] = np.nan
    table.loc[4, "emg1_mav"] = np.inf
    d = dashboard_data(table, features=["emg1_mav", "emg1_rms"], validate=False)
    assert d["values"]["emg1_mav"][3] is None and d["values"]["emg1_mav"][4] is None
    assert d["values"]["emg1_rms"][3] is not None
    json.dumps(d, allow_nan=False)  # raises on NaN or Infinity
    render_dashboard(d)


def test_a_feature_without_any_finite_value_is_left_out(cohort):
    table = cohort.assign(empty=np.nan)
    d = dashboard_data(table, validate=False)
    assert "empty" not in {f["name"] for f in d["features"]}


def test_rows_without_subject_or_label_are_left_out_and_counted(cohort):
    table = cohort.copy()
    table.loc[0, "subject"] = None
    table.loc[1, "label"] = None
    d = dashboard_data(table, validate=False)
    assert d["dropped_rows"] == 2 and d["n_rows"] == 46


def test_a_large_table_is_sampled_the_same_way_every_time(cohort):
    first = dashboard_data(cohort, max_rows=20, validate=False)
    again = dashboard_data(cohort, max_rows=20, validate=False)
    other = dashboard_data(cohort, max_rows=20, validate=False, seed=1)
    assert first["n_rows"] == 20 and first["sampled_from"] == 48
    assert first["values"] == again["values"] and first["rows"] == again["rows"]
    assert first["rows"]["id"] != other["rows"]["id"] or first["values"] != other["values"]


def test_labels_option_keeps_only_the_named_labels(cohort):
    d = dashboard_data(cohort, labels=["10%", "70%"], validate=False)
    assert d["labels"] == ["10%", "70%"] and d["n_rows"] == 24
    with pytest.raises(ValueError, match="none of the labels"):
        dashboard_data(cohort, labels=["99%"], validate=False)


def test_ids_name_the_column_they_come_from(cohort):
    assert dashboard_data(cohort, validate=False)["id_name"] == "trial"
    seg = cohort.drop(columns="trial").assign(segment_id=np.arange(100, 148))
    d = dashboard_data(seg, validate=False)
    assert d["id_name"] == "segment" and d["rows"]["id"][0] == "100"
    bare = cohort.drop(columns="trial")
    d = dashboard_data(bare, validate=False)
    assert d["id_name"] == "row" and d["rows"]["id"][:2] == ["1", "2"]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"subject_col": "nobody"}, "column 'nobody' not found"),
        ({"label_col": "nothing"}, "column 'nothing' not found"),
        ({"features": ["emg1_mav", "nope"]}, r"feature column\(s\) not found"),
        ({"features": ["session"]}, "not numeric"),
        ({"models": ("lda", "transformer")}, "unknown model"),
        ({"models": ()}, "models is empty"),
    ],
)
def test_bad_options_are_explained(cohort, options, message):
    with pytest.raises(ValueError, match=message):
        dashboard_data(cohort, **options)


def test_a_table_without_numeric_features_is_refused():
    with pytest.raises(ValueError, match="no numeric feature columns"):
        dashboard_data(pd.DataFrame({"subject": ["a", "b"], "label": ["x", "y"]}))


def test_more_labels_than_colours_is_refused_with_the_way_out():
    table = small_table([f"c{i}" for i in range(9)], reps=1)
    with pytest.raises(ValueError, match=r"9 labels.*labels=\[\.\.\.\]"):
        dashboard_data(table, validate=False)
    keep = [f"c{i}" for i in range(8)]
    assert len(dashboard_data(table, labels=keep, validate=False)["labels"]) == 8


# ------------------------------------------------------------------------ the validation
def test_validation_compares_three_schemes_for_each_model(cohort):
    v = dashboard_data(cohort, models=("lda", "svm_linear"))["validation"]
    assert [s["key"] for s in v["scenarios"]] == ["kfold|none", "loso|none", "loso|within_group"]
    assert [s["leaky"] for s in v["scenarios"]] == [True, False, False]
    assert [s["zscore"] for s in v["scenarios"]] == [False, False, True]
    assert [m["key"] for m in v["models"]] == ["lda", "svm_linear"]
    assert len(v["results"]) == 6
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in v["results"])
    assert v["baseline"] == 0.25 and v["baseline_name"] == "chance"
    assert v["n_rows"] == 48


def test_validation_of_unbalanced_classes_is_compared_with_the_majority_class():
    table = small_table(["a", "b"], n_subjects=4, reps=2)
    extra = pd.concat([table[table["label"] == "a"]] * 2, ignore_index=True)
    v = dashboard_data(pd.concat([table, extra], ignore_index=True), models=("lda",))["validation"]
    assert v["baseline_name"] == "majority class" and v["baseline"] == pytest.approx(0.75)


def test_validation_given_by_the_caller_is_used_as_it_is(cohort):
    names = [f["name"] for f in dashboard_data(cohort, validate=False)["features"]]
    results = compare_scenarios(cohort, names, "label", "subject", models=("lda",))
    d = dashboard_data(cohort, validation=results, models=())  # models are not needed then
    got = {(r["model"], r["scenario"]): r["accuracy"] for r in d["validation"]["results"]}
    for row in results.itertuples():
        assert got[(row.model, f"{row.scheme}|{row.normalization}")] == pytest.approx(
            row.accuracy, abs=1e-5
        )


def test_validation_with_more_than_three_scenarios_is_refused(cohort):
    names = [f["name"] for f in dashboard_data(cohort, validate=False)["features"]]
    scenarios = (
        ("kfold", "none"),
        ("loso", "none"),
        ("loso", "within_group"),
        ("kfold", "within_group"),
    )
    results = compare_scenarios(
        cohort, names, "label", "subject", models=("lda",), scenarios=scenarios
    )
    with pytest.raises(ValueError, match="at most three"):
        dashboard_data(cohort, validation=results)


def test_too_small_a_table_gets_a_reason_instead_of_a_chart():
    two = dashboard_data(small_table(["a", "b"], n_subjects=2))["validation"]
    assert "at least 3 subjects" in two["error"]
    one_label = dashboard_data(small_table(["a"], n_subjects=4))["validation"]
    assert "at least 2 labels" in one_label["error"]
    # and the rest of the page is still there
    assert dashboard_data(small_table(["a", "b"], n_subjects=2))["features"]


# --------------------------------------------------------------------------- the page
def test_the_page_carries_its_data_and_nothing_to_fetch(cohort):
    d = dashboard_data(cohort, title="Cohort", source="cohort.csv", models=("lda",))
    page = render_dashboard(d)
    assert page.startswith("<!DOCTYPE html>") and "<title>Cohort</title>" in page
    assert embedded_data(page) == json.loads(json.dumps(d))
    assert "{{TITLE}}" not in page and "{{DATA}}" not in page  # every slot was filled

    # nothing leaves the page: no links, images, imports or requests ...
    for needle in ("<link", "<img", "<iframe", "@import", "fetch(", "XMLHttpRequest", "import("):
        assert needle not in page, needle
    assert not re.search(r"<script[^>]*\ssrc=", page)
    assert not re.search(r"""\s(?:href|src)\s*=\s*["']https?:""", page)
    # ... and the only absolute URL is the SVG namespace the charts are drawn in
    assert set(re.findall(r"https?://[^\s\"'<>)]+", page)) == {"http://www.w3.org/2000/svg"}
    assert page.count("</script") == 2  # the data and the code, nothing else


def test_text_from_the_table_cannot_break_out_of_the_page():
    evil = '</script><img src=x onerror="alert(1)"> & {{DATA}} <!--'
    table = small_table([evil, "b"], n_subjects=3)
    table["subject"] = table["subject"].str.replace("P1", evil, regex=False)
    d = dashboard_data(table, title=evil, source=evil, validate=False)
    page = render_dashboard(d)
    assert page.count("</script") == 2 and "<img" not in page
    assert page.count('id="data"') == 1
    assert "<title>&lt;/script&gt;" in page and "{{DATA}}" in page.split("</title>")[0]
    assert embedded_data(page) == json.loads(json.dumps(d))  # the values survive the escaping
    assert evil in embedded_data(page)["labels"]


def test_the_template_has_one_slot_for_the_title_and_one_for_the_data():
    template = resources.files("biosigkit").joinpath("dashboard.html").read_text(encoding="utf-8")
    assert template.count("{{TITLE}}") == 1 and template.count("{{DATA}}") == 1
    assert "<title>{{TITLE}}</title>" in template


def test_write_dashboard_writes_utf8_with_unix_newlines(cohort, tmp_path):
    out = write_dashboard(cohort, tmp_path / "force.html", title="Força — ∂ 日本語", validate=False)
    raw = out.read_bytes()
    text = raw.decode("utf-8")  # raises if it was written in the platform's default encoding
    assert b"\r" not in raw
    assert "<title>Força — ∂ 日本語</title>" in text
    assert embedded_data(text)["title"] == "Força — ∂ 日本語"


def test_the_page_keeps_the_promises_it_makes_in_its_markup():
    template = resources.files("biosigkit").joinpath("dashboard.html").read_text(encoding="utf-8")
    ids = re.findall(r'\sid="([^"]+)"', template)
    assert len(ids) == len(set(ids)), "duplicate ids in the template"
    # every toggle points at a table region that exists
    for target in re.findall(r'aria-controls="([^"]+)"', template):
        assert f'id="{target}"' in template
    # the colour themes are declared in both scopes: the system setting and the toggle
    assert "prefers-color-scheme: dark" in template and '[data-theme="dark"]' in template
    assert "forced-colors" in template


# ---------------------------------------------------------------------- the live demo page
DEMO_PAGE = Path(__file__).resolve().parents[1] / "docs" / "dashboard-demo.html"


def test_the_live_demo_page_is_a_dashboard_of_the_demo_cohort():
    if not DEMO_PAGE.exists():  # e.g. tests run against an installed wheel
        pytest.skip("docs/ is not available")
    page = DEMO_PAGE.read_text(encoding="utf-8")
    data = embedded_data(page)
    assert data["n_rows"] == 192 and len(data["subjects"]) == 8
    assert data["labels"] == ["10%", "30%", "50%", "70%"]
    assert data["validation"]["baseline"] == 0.25 and len(data["validation"]["results"]) == 12
    assert set(re.findall(r"https?://[^\s\"'<>)]+", page)) == {"http://www.w3.org/2000/svg"}
    assert b"\r" not in DEMO_PAGE.read_bytes()


def test_the_live_demo_page_was_built_from_the_current_template():
    if not DEMO_PAGE.exists():
        pytest.skip("docs/ is not available")
    template = resources.files("biosigkit").joinpath("dashboard.html").read_text(encoding="utf-8")
    pattern = re.escape(template)
    for slot in ("{{TITLE}}", "{{DATA}}"):
        pattern = pattern.replace(re.escape(slot), "(?:.*?)")
    page = DEMO_PAGE.read_text(encoding="utf-8")
    assert re.fullmatch(pattern, page, re.S), (
        "docs/dashboard-demo.html is out of date: run python examples/build_dashboard_demo.py"
    )
