"""Leakage-aware validation: the guarantees that make the reported accuracy honest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from biosigkit.ml import (
    DEFAULT_SCENARIOS,
    MODEL_NAMES,
    compare_scenarios,
    cross_validate,
    make_model,
    within_group_zscore,
)

FEATURES = ["x0", "x1", "x2"]


def subject_signature_data(n_subjects: int = 10, n_rows: int = 30, seed: int = 0) -> pd.DataFrame:
    """Features identify the *subject* and nothing else; labels are biased per subject.

    Every subject has a huge constant offset ("electrode placement") and a favourite class that
    80 % of their rows share. A model that memorises the offset looks excellent under a random
    split and is useless on a new subject.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        signature = rng.normal(0.0, 4.0, size=3)
        favourite = s % 3
        for _ in range(n_rows):
            label = favourite if rng.random() < 0.8 else int(rng.integers(3))
            x = signature + rng.normal(0.0, 1.0, size=3)
            rows.append(
                {
                    "subject": f"S{s:02d}",
                    "label": f"c{label}",
                    **dict(zip(FEATURES, x, strict=True)),
                }
            )
    return pd.DataFrame(rows)


def subject_gain_data(n_subjects: int = 8, n_rows: int = 20, seed: int = 1) -> pd.DataFrame:
    """Class information is real, but every subject scales all features by their own gain."""
    rng = np.random.default_rng(seed)
    centres = np.array([[1.0, 1.0, 1.0], [2.0, 1.5, 2.5], [3.0, 2.0, 4.0]])
    rows = []
    for s in range(n_subjects):
        gain = float(np.exp(rng.normal(0.0, 0.8)))
        for c in range(3):
            for _ in range(n_rows):
                x = gain * (centres[c] + rng.normal(0.0, 0.15, size=3))
                rows.append(
                    {
                        "subject": f"S{s:02d}",
                        "label": f"c{c}",
                        **dict(zip(FEATURES, x, strict=True)),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 48  # 4 subjects x 12 rows
    return pd.DataFrame(
        {
            "subject": np.repeat(["A", "B", "C", "D"], n // 4),
            "label": np.tile(["a", "b", "c"], n // 3),
            **{f: rng.normal(size=n) for f in FEATURES},
        }
    )


# --------------------------------------------------------------------------- the main point
def test_random_kfold_is_optimistic_and_loso_is_not():
    df = subject_signature_data()
    leaky = cross_validate(df, FEATURES, "label", "subject", model="svm_rbf", scheme="kfold")
    honest = cross_validate(df, FEATURES, "label", "subject", model="svm_rbf", scheme="loso")
    assert leaky.accuracy > 0.75  # "recognises the subject"
    assert honest.accuracy < 0.5  # chance is 0.33: there is nothing to learn across subjects
    assert leaky.accuracy - honest.accuracy > 0.3


def test_within_subject_normalisation_removes_between_subject_gain():
    df = subject_gain_data()
    res = compare_scenarios(
        df,
        FEATURES,
        "label",
        "subject",
        models=("lda",),
        scenarios=[("loso", "none"), ("loso", "within_group")],
    ).set_index("normalization")
    assert res.loc["within_group", "accuracy"] > 0.85
    assert res.loc["within_group", "accuracy"] > res.loc["none", "accuracy"] + 0.2


# ----------------------------------------------------------------------- split invariants
@pytest.mark.parametrize("scheme", ["loso", "group_kfold", "kfold"])
def test_every_row_is_predicted_exactly_once(small_df, scheme):
    res = cross_validate(small_df, FEATURES, "label", "subject", scheme=scheme, n_splits=4)
    assert res.predictions["row"].sort_values().tolist() == list(range(len(small_df)))


def test_loso_holds_out_exactly_one_subject_per_fold(small_df):
    res = cross_validate(small_df, FEATURES, "label", "subject", scheme="loso")
    assert len(res.folds) == small_df["subject"].nunique()
    assert sorted(res.folds["held_out"]) == ["A", "B", "C", "D"]
    assert (res.predictions.groupby("fold")["group"].nunique() == 1).all()
    assert (res.folds["n_train"] + res.folds["n_test"] == len(small_df)).all()


def test_group_kfold_never_splits_a_subject_across_folds(small_df):
    res = cross_validate(small_df, FEATURES, "label", "subject", scheme="group_kfold", n_splits=2)
    subjects_per_fold = res.predictions.groupby("fold")["group"].agg(set)
    assert sum(len(s) for s in subjects_per_fold) == small_df["subject"].nunique()


def test_results_are_reproducible(small_df):
    kwargs = {"model": "rf", "scheme": "kfold", "n_splits": 2}  # the random forest is stochastic
    a = cross_validate(small_df, FEATURES, "label", "subject", **kwargs)
    b = cross_validate(small_df, FEATURES, "label", "subject", **kwargs)
    pd.testing.assert_frame_equal(a.predictions, b.predictions)


def test_result_metrics_are_consistent(small_df):
    res = cross_validate(small_df, FEATURES, "label", "subject", scheme="loso")
    cm = res.confusion()
    assert list(cm.index) == list(cm.columns) == ["a", "b", "c"]
    assert cm.to_numpy().sum() == len(small_df)
    assert np.trace(cm.to_numpy()) / len(small_df) == pytest.approx(res.accuracy)
    summary = res.summary()
    assert summary["n_folds"] == 4
    assert all(0.0 <= summary[k] <= 1.0 for k in ("accuracy", "balanced_accuracy", "f1_macro"))


def test_cross_validate_rejects_bad_input(small_df):
    with pytest.raises(ValueError, match="at least 2 groups"):
        cross_validate(small_df[small_df["subject"] == "A"], FEATURES, "label", "subject")
    broken = small_df.copy()
    broken.loc[0, "x0"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        cross_validate(broken, FEATURES, "label", "subject")
    with pytest.raises(ValueError, match="scheme"):
        cross_validate(small_df, FEATURES, "label", "subject", scheme="bogus")  # type: ignore[arg-type]


# ------------------------------------------------------------------------ normalisation
def test_within_group_zscore_standardises_every_group(rng):
    df = pd.DataFrame(
        {
            "subject": np.repeat(["a", "b", "c"], 20),
            "x": np.concatenate(
                [rng.normal(5, 2, 20), rng.normal(-3, 0.1, 20), rng.normal(100, 30, 20)]
            ),
            "const": 7.0,
        }
    )
    before = df.copy()
    out = within_group_zscore(df, ["x", "const"], "subject")
    grouped = out.groupby("subject")["x"]
    assert grouped.mean().to_numpy() == pytest.approx(0.0, abs=1e-12)
    assert grouped.std(ddof=0).to_numpy() == pytest.approx(1.0)
    assert (out["const"] == 0).all()  # zero variance -> 0, not NaN
    pd.testing.assert_frame_equal(df, before)  # the input is not modified


def test_within_group_zscore_keeps_missing_values_missing():
    df = pd.DataFrame({"g": ["a"] * 4, "x": [1.0, 2.0, np.nan, 4.0]})
    out = within_group_zscore(df, ["x"], "g")
    assert np.isnan(out.loc[2, "x"])
    assert out["x"].notna().sum() == 3


# ----------------------------------------------------------------- models and scenarios
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_are_pipelines_that_scale_inside_the_fold(name):
    model = make_model(name)
    assert isinstance(model, Pipeline)
    assert model.steps[0][0] == "scale"


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        make_model("transformer")


def test_compare_scenarios_returns_one_tidy_row_per_combination(small_df):
    res = compare_scenarios(small_df, FEATURES, "label", "subject", models=("lda", "logreg"))
    assert len(res) == len(DEFAULT_SCENARIOS) * 2
    assert list(res.columns[:3]) == ["model", "scheme", "normalization"]
    assert res["accuracy"].between(0, 1).all()
    with pytest.raises(ValueError, match="normalization"):
        compare_scenarios(small_df, FEATURES, "label", "subject", scenarios=[("loso", "rank")])
