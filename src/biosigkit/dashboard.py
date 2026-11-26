"""A self-contained HTML dashboard for a trial-level feature table.

``write_dashboard(table, "dashboard.html")`` writes one file: no server, no network, no extra
dependency. Open it from disk and explore one feature at a time, by subject and by label, with and
without a per-subject z-score, next to the accuracy each validation scheme gives. Every chart has a
table view, the filters work from the keyboard, and the page follows the system light/dark setting.

The table needs one row per trial (or window), a subject column, a label column and numeric
feature columns, i.e. what ``trial_feature_table`` or ``biosigkit features --mode trial`` writes.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .ml import DEFAULT_SCENARIOS, MODEL_NAMES, compare_scenarios, within_group_zscore
from .tables import RESERVED_COLUMNS, infer_feature_columns

MAX_LABELS = 8
MAX_ORDINAL_LABELS = 4  # the one-hue lightness ramp that passes the colour checks in both themes
LabelScale = Literal["auto", "ordinal", "nominal"]

MODEL_TITLES = {
    "lda": "LDA",
    "svm_rbf": "SVM (RBF)",
    "svm_linear": "SVM (linear)",
    "rf": "Random forest",
    "logreg": "Logistic regression",
}
SCENARIO_TITLES = {
    ("kfold", "none"): "Random split",
    ("loso", "none"): "Leave-one-subject-out",
    ("loso", "within_group"): "Leave-one-subject-out + z-score per subject",
    ("group_kfold", "none"): "Group k-fold",
    ("group_kfold", "within_group"): "Group k-fold + z-score per subject",
    ("kfold", "within_group"): "Random split + z-score per subject",
}
_NUMERIC_LABEL = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*(?:%|[A-Za-z]{1,4})?\s*$")


def _natural_key(text: str) -> list[Any]:
    """Sort key that puts ``S2`` before ``S10`` (digit runs compare as numbers)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _number(value: float) -> float | None:
    """Finite floats to 6 significant digits, anything else to ``None`` (JSON ``null``)."""
    return float(f"{value:.6g}") if np.isfinite(value) else None


def _eta_squared(values: np.ndarray, codes: np.ndarray) -> float:
    """Share of the variance of ``values`` explained by the group ``codes`` (0 to 1)."""
    ok = np.isfinite(values)
    x, g = values[ok], codes[ok].astype(np.intp)
    if x.size < 3 or np.unique(g).size < 2:
        return float("nan")
    total = float(((x - x.mean()) ** 2).sum())
    if total <= 0:
        return float("nan")
    counts = np.bincount(g)
    means = np.bincount(g, weights=x) / np.where(counts > 0, counts, 1)
    between = float((counts * (means - x.mean()) ** 2).sum())
    return between / total


def _order_labels(labels: list[str], scale: LabelScale) -> tuple[list[str], str]:
    """Order the labels and decide how they are coloured: a lightness ramp or distinct hues."""
    numeric = [_NUMERIC_LABEL.match(s) for s in labels]
    all_numeric = all(m is not None for m in numeric)
    if scale == "nominal" or (scale == "auto" and not all_numeric):
        return labels, "nominal"  # keep the order of the file
    if all_numeric:
        values = {s: float(m.group(1)) for s, m in zip(labels, numeric, strict=True) if m}
        ordered = sorted(labels, key=lambda s: (values[s], s))
    else:  # asked for an ordinal scale, but the labels are not numbers
        ordered = sorted(labels, key=_natural_key)
    ramp = 3 <= len(ordered) <= MAX_ORDINAL_LABELS
    return ordered, "ordinal" if ramp else "nominal"


def _validation_payload(
    results: pd.DataFrame, n_labels: int, majority: float, n_rows: int
) -> dict[str, Any]:
    needed = {"model", "scheme", "normalization", "accuracy"}
    if not needed <= set(results.columns):
        raise ValueError(f"validation results need the columns {sorted(needed)}")
    scenarios: list[tuple[str, str]] = []
    for scheme, norm in zip(results["scheme"], results["normalization"], strict=True):
        if (scheme, norm) not in scenarios:
            scenarios.append((scheme, norm))
    if len(scenarios) > 3:
        raise ValueError("the validation chart shows at most three scenarios")
    models = list(dict.fromkeys(results["model"]))
    balanced = abs(majority - 1 / n_labels) <= 0.005
    return {
        "scenarios": [
            {
                "key": f"{scheme}|{norm}",
                "name": SCENARIO_TITLES.get((scheme, norm), f"{scheme}, {norm}"),
                "leaky": scheme == "kfold",  # a random split lets a subject sit on both sides
                "zscore": norm == "within_group",
            }
            for scheme, norm in scenarios
        ],
        "models": [{"key": m, "name": MODEL_TITLES.get(m, m)} for m in models],
        "results": [
            {
                "model": r["model"],
                "scenario": f"{r['scheme']}|{r['normalization']}",
                "accuracy": _number(float(r["accuracy"])),
                "balanced_accuracy": _number(float(r.get("balanced_accuracy", np.nan))),
            }
            for r in results.to_dict("records")
        ],
        "baseline": _number(majority),
        "baseline_name": "chance" if balanced else "majority class",
        "n_rows": n_rows,
    }


def _run_validation(
    work: pd.DataFrame,
    feats: list[str],
    subject_col: str,
    label_col: str,
    models: Sequence[str],
) -> pd.DataFrame:
    complete = work.dropna(subset=feats)
    if complete[subject_col].nunique() < 3:
        raise ValueError("validation needs at least 3 subjects with complete rows")
    counts = complete[label_col].value_counts()
    if counts.size < 2 or counts.min() < 2:
        raise ValueError("validation needs at least 2 labels with 2 or more rows each")
    n_splits = int(min(5, counts.min()))
    return compare_scenarios(
        complete,
        feats,
        label_col,
        subject_col,
        models=models,
        scenarios=DEFAULT_SCENARIOS,
        n_splits=n_splits,
    )


def dashboard_data(
    table: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    subject_col: str = "subject",
    label_col: str = "label",
    labels: Sequence[str] | None = None,
    label_scale: LabelScale = "auto",
    models: Sequence[str] = ("lda", "svm_rbf", "svm_linear", "rf"),
    validate: bool = True,
    validation: pd.DataFrame | None = None,
    title: str = "Feature dashboard",
    source: str | None = None,
    max_rows: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """The JSON-serialisable payload the page draws from.

    ``features`` defaults to every numeric column that is not bookkeeping (subject, label, times).
    ``labels`` keeps only those labels; at most 8 can be drawn. ``validation`` takes the output of
    :func:`biosigkit.ml.compare_scenarios` (at most three scenarios) so a run that already has it
    is not repeated; otherwise it is computed for ``models`` unless ``validate`` is false or the
    table is too small, in which case the page says why. A table with more than ``max_rows`` rows
    is sampled at random (``seed``), and the page says so.
    """
    from . import __version__  # imported here: the package defines it after importing this module

    for col in (subject_col, label_col):
        if col not in table.columns:
            raise ValueError(f"column {col!r} not found; columns are {list(table.columns)}")
    unknown = [m for m in models if m not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}; choose from {list(MODEL_NAMES)}")
    if validate and validation is None and not models:
        raise ValueError("models is empty; name at least one model or turn the validation off")
    if features is None:
        feats = infer_feature_columns(table, RESERVED_COLUMNS | {subject_col, label_col})
    else:
        feats = list(features)
        missing = [c for c in feats if c not in table.columns]
        if missing:
            raise ValueError(f"feature column(s) not found: {missing}")
        bad = [c for c in feats if not pd.api.types.is_numeric_dtype(table[c])]
        if bad:
            raise ValueError(f"feature column(s) are not numeric: {bad}")
    feats = [
        c
        for c in feats
        if c not in (subject_col, label_col) and np.isfinite(table[c].to_numpy(dtype=float)).any()
    ]
    if not feats:
        raise ValueError("no numeric feature columns with values were found")

    work = table[[subject_col, label_col, *feats]].copy()
    work["_row"] = np.arange(len(work)) + 1
    if "trial" in table.columns:
        id_name, work["_id"] = "trial", table["trial"].astype(str)
    elif "segment_id" in table.columns:
        id_name, work["_id"] = "segment", table["segment_id"].astype(str)
    else:
        id_name, work["_id"] = "row", work["_row"].astype(str)
    n_before = len(work)
    work = work.dropna(subset=[subject_col, label_col])
    dropped = n_before - len(work)
    work[subject_col] = work[subject_col].astype(str)
    work[label_col] = work[label_col].astype(str)
    if labels is not None:
        keep = [str(x) for x in labels]
        work = work[work[label_col].isin(keep)]
        if work.empty:
            raise ValueError(f"none of the labels {keep} is in the table")
    if work.empty:
        raise ValueError("the table has no rows with a subject and a label")
    sampled_from = None
    if len(work) > max_rows:
        sampled_from = len(work)
        work = work.sample(n=max_rows, random_state=seed).sort_index()
    work = work.reset_index(drop=True)

    label_list, scale = _order_labels(list(dict.fromkeys(work[label_col])), label_scale)
    if len(label_list) > MAX_LABELS:
        raise ValueError(
            f"{len(label_list)} labels: the dashboard colours at most {MAX_LABELS}; "
            "pick some with labels=[...] (--labels)"
        )
    subjects = sorted(dict.fromkeys(work[subject_col]), key=_natural_key)
    subject_codes = work[subject_col].map({s: i for i, s in enumerate(subjects)}).to_numpy()
    label_codes = work[label_col].map({s: i for i, s in enumerate(label_list)}).to_numpy()

    zscored = within_group_zscore(work, feats, subject_col)
    meta: list[dict[str, Any]] = []
    for name in feats:
        meta.append(
            {
                "name": name,
                "eta_raw": _number(_eta_squared(work[name].to_numpy(float), label_codes)),
                "eta_z": _number(_eta_squared(zscored[name].to_numpy(float), label_codes)),
            }
        )
    meta.sort(key=lambda m: (-(m["eta_z"] if m["eta_z"] is not None else -1.0), m["name"]))

    validation_payload: dict[str, Any] | None = None
    if validation is not None or validate:
        counts = work[label_col].value_counts(normalize=True)
        majority = float(counts.max())
        try:
            results = (
                validation
                if validation is not None
                else _run_validation(work, feats, subject_col, label_col, models)
            )
            validation_payload = _validation_payload(
                results, len(label_list), majority, int(len(work.dropna(subset=feats)))
            )
        except ValueError as exc:
            if validation is not None:
                raise
            validation_payload = {"error": str(exc)}

    return {
        "title": title,
        "source": source,
        "version": __version__,
        "n_rows": int(len(work)),
        "sampled_from": sampled_from,
        "dropped_rows": int(dropped),
        "subjects": subjects,
        "labels": label_list,
        "label_scale": scale,
        "features": meta,
        "rows": {
            "subject": [int(c) for c in subject_codes],
            "label": [int(c) for c in label_codes],
            "id": work["_id"].tolist(),
        },
        "id_name": id_name,
        "values": {name: [_number(float(v)) for v in work[name].to_numpy(float)] for name in feats},
        "validation": validation_payload,
    }


def render_dashboard(data: dict[str, Any]) -> str:
    """The page as a string: the template with the title and the data filled in."""
    template = resources.files("biosigkit").joinpath("dashboard.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    # nothing in the payload may close the <script> element or open a comment
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    values = {"TITLE": html.escape(str(data["title"])), "DATA": payload}
    return re.sub(r"\{\{(TITLE|DATA)\}\}", lambda m: values[m.group(1)], template)


def write_dashboard(table: pd.DataFrame, path: str | Path, **options: Any) -> Path:
    """Write the dashboard of ``table`` to ``path`` (UTF-8, LF line endings) and return the path."""
    out = Path(path)
    out.write_text(
        render_dashboard(dashboard_data(table, **options)), encoding="utf-8", newline="\n"
    )
    return out
