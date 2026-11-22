"""Tidy feature tables: windows or trials -> DataFrame, long/wide conversion and summaries.

Column convention: feature columns are named ``<channel>_<feature>`` (e.g. ``emg1_rms``). Feature
names never contain an underscore, so the pair can always be recovered with ``rsplit("_", 1)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from .emg import emg_features
from .signals import Signal
from .windowing import sliding_windows, validate_events

SEP = "_"
FeatureFn = Callable[[np.ndarray, float], Mapping[str, np.ndarray]]
RESERVED_COLUMNS = frozenset(
    {"subject", "session", "trial", "segment_id", "kind", "t_start", "t_stop", "label", "level"}
)


def _feature_frame(features: Mapping[str, np.ndarray], channels: Sequence[str]) -> pd.DataFrame:
    """``{feature: (n, n_channels)}`` -> DataFrame with ``<channel>_<feature>`` columns."""
    columns: dict[str, np.ndarray] = {}
    for idx, channel in enumerate(channels):
        for name, values in features.items():
            if SEP in name:
                raise ValueError(f"feature name {name!r} must not contain {SEP!r}")
            columns[f"{channel}{SEP}{name}"] = np.asarray(values, dtype=float)[:, idx]
    return pd.DataFrame(columns)


def _with_meta(table: pd.DataFrame, meta: Mapping[str, Any] | None) -> pd.DataFrame:
    for pos, (key, value) in enumerate((meta or {}).items()):
        table.insert(pos, key, value)
    return table


def label_by_overlap(
    t_start: np.ndarray,
    t_stop: np.ndarray,
    events: pd.DataFrame,
    *,
    rest_label: str = "rest",
    min_overlap: float = 0.5,
) -> np.ndarray:
    """Label each window with the event it overlaps most (``rest_label`` if < ``min_overlap``).

    ``min_overlap`` is the fraction of the *window* that must lie inside the event.
    """
    t_start, t_stop = np.asarray(t_start, dtype=float), np.asarray(t_stop, dtype=float)
    ev = validate_events(events)
    if ev.empty:
        return np.full(t_start.shape, rest_label, dtype=object)
    onset, offset = ev["onset"].to_numpy(), ev["offset"].to_numpy()
    overlap = np.minimum(t_stop[:, None], offset[None, :]) - np.maximum(
        t_start[:, None], onset[None, :]
    )
    overlap = np.clip(overlap, 0.0, None)
    best = overlap.argmax(axis=1)
    fraction = overlap[np.arange(len(t_start)), best] / (t_stop - t_start)
    labels = ev["label"].to_numpy(dtype=object)[best]
    return np.where(fraction >= min_overlap - 1e-9, labels, rest_label).astype(object)


def window_feature_table(
    sig: Signal,
    *,
    win_s: float = 0.2,
    step_s: float = 0.1,
    feature_fn: FeatureFn | None = None,
    events: pd.DataFrame | None = None,
    rest_label: str = "rest",
    min_overlap: float = 0.5,
    meta: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """One row per sliding window: ``t_start, t_stop, [label,] <channel>_<feature>...``.

    ``feature_fn(windows, fs)`` must return ``{name: (n_windows, n_channels)}``; it defaults to
    :func:`biosigkit.emg.emg_features`. ``meta`` columns (e.g. ``{"subject": "S01"}``) are
    prepended.
    """
    fn = feature_fn or emg_features
    windows, t_start, t_stop = sliding_windows(sig, win_s, step_s)
    table = pd.DataFrame({"t_start": t_start, "t_stop": t_stop})
    if events is not None:
        table["label"] = label_by_overlap(
            t_start, t_stop, events, rest_label=rest_label, min_overlap=min_overlap
        )
    table = pd.concat([table, _feature_frame(fn(windows, sig.fs), sig.channels)], axis=1)
    return _with_meta(table, meta)


def trial_feature_table(
    sig: Signal,
    events: pd.DataFrame,
    *,
    feature_fn: FeatureFn | None = None,
    trim_s: float = 0.0,
    on_invalid: Literal["raise", "skip"] = "raise",
    meta: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """One row per event (trial): features over ``[onset + trim_s, offset - trim_s]``.

    Extra event columns (e.g. ``level``) are carried into the table. Use ``trim_s`` to drop ramp
    phases at the edges of a trial.
    """
    fn = feature_fn or emg_features
    ev = validate_events(events)
    extra = [c for c in ev.columns if c not in ("onset", "offset", "label")]
    info: list[dict[str, Any]] = []
    per_trial: list[Mapping[str, np.ndarray]] = []
    for i, rec in enumerate(ev.to_dict("records")):
        a = round((rec["onset"] + trim_s - sig.t0) * sig.fs)
        b = round((rec["offset"] - trim_s - sig.t0) * sig.fs)
        if a < 0 or b > sig.n_samples or b - a < 2:
            if on_invalid == "raise":
                raise ValueError(
                    f"trial {i} ({rec['onset']}-{rec['offset']} s) lies outside the recording "
                    "or is too short after trimming"
                )
            continue
        feats = fn(sig.data[None, :, a:b], sig.fs)
        per_trial.append({k: np.asarray(v, dtype=float)[0] for k, v in feats.items()})
        info.append(
            {
                "trial": i,
                "t_start": sig.t0 + a / sig.fs,
                "t_stop": sig.t0 + b / sig.fs,
                "label": rec["label"],
                **{k: rec[k] for k in extra},
            }
        )
    if not info:
        raise ValueError("no valid trials found")
    stacked = {k: np.stack([t[k] for t in per_trial]) for k in per_trial[0]}
    table = pd.concat([pd.DataFrame(info), _feature_frame(stacked, sig.channels)], axis=1)
    return _with_meta(table, meta)


def infer_feature_columns(
    df: pd.DataFrame, exclude: Sequence[str] | frozenset[str] = RESERVED_COLUMNS
) -> list[str]:
    """Numeric columns that are not bookkeeping columns (subject, label, times, ...)."""
    skip = set(exclude)
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def features_to_long(
    df: pd.DataFrame, id_cols: Sequence[str], feature_cols: Sequence[str] | None = None
) -> pd.DataFrame:
    """Wide ``<channel>_<feature>`` columns -> long ``id_cols + [channel, feature, value]``."""
    cols = list(feature_cols) if feature_cols is not None else infer_feature_columns(df)
    long = df.melt(id_vars=list(id_cols), value_vars=cols, var_name="_col", value_name="value")
    parts = long["_col"].str.rsplit(SEP, n=1, expand=True)
    if parts.shape[1] != 2 or parts.isna().any().any():
        raise ValueError(f"feature columns must look like '<channel>{SEP}<feature>'")
    long["channel"], long["feature"] = parts[0], parts[1]
    return long.drop(columns="_col")


def features_to_wide(long: pd.DataFrame, index_col: str = "segment_id") -> pd.DataFrame:
    """Inverse of :func:`features_to_long` for the feature part (one row per ``index_col``)."""
    col = long["channel"] + SEP + long["feature"]
    wide = long.assign(_col=col).pivot(index=index_col, columns="_col", values="value")
    wide.columns.name = None
    return wide[list(dict.fromkeys(col))]


def summarize(
    df: pd.DataFrame,
    by: Sequence[str],
    value_cols: Sequence[str] | None = None,
    stats: Sequence[str] = ("mean", "std", "count"),
) -> pd.DataFrame:
    """Group statistics with flat ``<column>.<stat>`` names (handy for reports and CSV export)."""
    cols = (
        list(value_cols)
        if value_cols is not None
        else infer_feature_columns(df, RESERVED_COLUMNS | set(by))
    )
    out = df.groupby(list(by), observed=True)[cols].agg(list(stats))
    out.columns = [f"{col}.{stat}" for col, stat in out.columns]
    return out.reset_index()


def validate_table(
    df: pd.DataFrame,
    *,
    required: Sequence[str] = (),
    finite: Sequence[str] | None = None,
) -> None:
    """Raise ``ValueError`` if columns are missing or feature values are NaN/inf."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"table is missing required columns: {missing}")
    cols = list(finite) if finite is not None else infer_feature_columns(df)
    if cols:
        bad = ~np.isfinite(df[cols].to_numpy(dtype=float))
        if bad.any():
            names = [c for c, flag in zip(cols, bad.any(axis=0), strict=True) if flag]
            raise ValueError(f"non-finite values in columns: {names}")
