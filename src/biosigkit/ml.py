"""Leakage-aware model evaluation for data with a subject (group) structure.

Rows that come from the same subject are *not* independent: they share electrode placement, skin
impedance, anatomy, effort level, ... A random train/test split therefore lets the model "recognise
the subject" and reports an optimistic accuracy. Evaluating with whole subjects held out
(leave-one-subject-out, LOSO) measures what actually matters for a new user.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_NAMES = ("lda", "svm_rbf", "svm_linear", "rf", "logreg")
Scheme = Literal["loso", "group_kfold", "kfold"]
DEFAULT_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("kfold", "none"),  # naive random split: leaks subject identity
    ("loso", "none"),  # honest estimate
    ("loso", "within_group"),  # honest estimate + per-subject normalisation
)


def make_model(name: str, random_state: int = 0) -> Pipeline:
    """Scaler + classifier pipeline. The scaler is fitted on the training fold only."""
    if name == "lda":
        clf = LinearDiscriminantAnalysis()
    elif name == "svm_rbf":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)
    elif name == "svm_linear":
        clf = SVC(kernel="linear", C=1.0, random_state=random_state)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    elif name == "logreg":
        clf = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def within_group_zscore(
    df: pd.DataFrame, feature_cols: Sequence[str], group_col: str, ddof: int = 0
) -> pd.DataFrame:
    """Z-score each feature inside every group (e.g. subject). Returns a modified copy.

    This removes between-subject differences in scale (electrode gain, skin impedance, ...) that
    often dominate the feature space. Caveat: the statistics of a held-out subject are computed
    from that subject's own (unlabelled) rows, so deployment needs a short calibration recording
    for every new user. Zero-variance columns become 0.
    """
    cols = list(feature_cols)
    out = df.copy()
    grouped = out.groupby(group_col)[cols]
    x = out[cols].to_numpy(dtype=float)
    mean = grouped.transform("mean").to_numpy(dtype=float)
    std = grouped.transform("std", ddof=ddof).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(std > 0, (x - mean) / std, 0.0)
    z[np.isnan(x)] = np.nan
    out[cols] = z
    return out


@dataclass
class CVResult:
    """Outcome of one cross-validation run."""

    scheme: str
    model: str
    folds: pd.DataFrame
    predictions: pd.DataFrame  # columns: row, group, fold, y_true, y_pred

    @property
    def labels(self) -> list[str]:
        return sorted(set(self.predictions["y_true"]) | set(self.predictions["y_pred"]))

    @property
    def accuracy(self) -> float:
        p = self.predictions
        return float(accuracy_score(p["y_true"], p["y_pred"]))

    @property
    def balanced_accuracy(self) -> float:
        p = self.predictions
        return float(balanced_accuracy_score(p["y_true"], p["y_pred"]))

    @property
    def f1_macro(self) -> float:
        p = self.predictions
        return float(f1_score(p["y_true"], p["y_pred"], average="macro", zero_division=0))

    def confusion(self) -> pd.DataFrame:
        """Pooled confusion matrix (rows: true label, columns: predicted label)."""
        labels = self.labels
        p = self.predictions
        cm = confusion_matrix(p["y_true"], p["y_pred"], labels=labels)
        return pd.DataFrame(cm, index=pd.Index(labels, name="true"), columns=labels)

    def summary(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1_macro": self.f1_macro,
            "fold_accuracy_mean": float(self.folds["accuracy"].mean()),
            "fold_accuracy_std": float(self.folds["accuracy"].std(ddof=0)),
            "n_folds": float(len(self.folds)),
        }


def cross_validate(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    group_col: str,
    *,
    model: str = "lda",
    scheme: Scheme = "loso",
    n_splits: int = 5,
    random_state: int = 0,
) -> CVResult:
    """Cross-validate a classifier on rows that belong to groups (subjects).

    Schemes
    -------
    ``"loso"``
        Leave-one-group-out: every fold holds out one whole subject.
    ``"group_kfold"``
        ``n_splits`` folds, groups never split across train and test.
    ``"kfold"``
        Shuffled stratified k-fold *ignoring* groups. Provided only to quantify the optimism of
        naive validation; do not use it to report performance.

    For the grouped schemes the function asserts that no group appears in both train and test.
    """
    cols = list(feature_cols)
    x = df[cols].to_numpy(dtype=float)
    y = df[label_col].to_numpy(dtype=object)
    groups = df[group_col].to_numpy(dtype=object)
    if not np.isfinite(x).all():
        raise ValueError("features contain NaN/inf; impute or drop those rows first")
    n_groups = len(set(groups))
    if scheme == "loso":
        if n_groups < 2:
            raise ValueError("LOSO needs at least 2 groups")
        splits = LeaveOneGroupOut().split(x, y, groups)
    elif scheme == "group_kfold":
        if n_groups < 2:
            raise ValueError("group_kfold needs at least 2 groups")
        splits = GroupKFold(n_splits=min(n_splits, n_groups)).split(x, y, groups)
    elif scheme == "kfold":
        splits = StratifiedKFold(n_splits, shuffle=True, random_state=random_state).split(x, y)
    else:
        raise ValueError("scheme must be 'loso', 'group_kfold' or 'kfold'")

    fold_rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for fold, (train, test) in enumerate(splits):
        shared = set(groups[train]) & set(groups[test])
        if scheme != "kfold" and shared:
            raise AssertionError(f"group leakage between train and test: {sorted(shared)}")
        pipe = make_model(model, random_state).fit(x[train], y[train])
        pred = pipe.predict(x[test])
        with warnings.catch_warnings():
            # A held-out subject may lack a class that the model predicts; the per-fold metric
            # then simply ignores that class, which is expected in LOSO and not worth a warning.
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            fold_rows.append(
                {
                    "fold": fold,
                    "held_out": ",".join(sorted(map(str, set(groups[test]))))
                    if scheme != "kfold"
                    else "-",
                    "n_train": len(train),
                    "n_test": len(test),
                    "accuracy": accuracy_score(y[test], pred),
                    "balanced_accuracy": balanced_accuracy_score(y[test], pred),
                    "f1_macro": f1_score(y[test], pred, average="macro", zero_division=0),
                }
            )
        predictions.append(
            pd.DataFrame(
                {
                    "row": test,
                    "group": groups[test],
                    "fold": fold,
                    "y_true": y[test],
                    "y_pred": pred,
                }
            )
        )
    return CVResult(
        scheme=scheme,
        model=model,
        folds=pd.DataFrame(fold_rows),
        predictions=pd.concat(predictions, ignore_index=True),
    )


def compare_scenarios(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    group_col: str,
    *,
    models: Sequence[str] = ("lda", "svm_rbf", "svm_linear", "rf"),
    scenarios: Sequence[tuple[str, str]] = DEFAULT_SCENARIOS,
    n_splits: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Run every model under every ``(scheme, normalization)`` scenario; one tidy row each.

    ``normalization`` is ``"none"`` or ``"within_group"`` (see :func:`within_group_zscore`).
    """
    cols = list(feature_cols)
    rows = []
    for scheme, normalization in scenarios:
        if normalization == "none":
            data = df
        elif normalization == "within_group":
            data = within_group_zscore(df, cols, group_col)
        else:
            raise ValueError("normalization must be 'none' or 'within_group'")
        for name in models:
            result = cross_validate(
                data,
                cols,
                label_col,
                group_col,
                model=name,
                scheme=scheme,  # type: ignore[arg-type]
                n_splits=n_splits,
                random_state=random_state,
            )
            rows.append(
                {
                    "model": name,
                    "scheme": scheme,
                    "normalization": normalization,
                    **result.summary(),
                }
            )
    return pd.DataFrame(rows)
