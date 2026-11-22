"""Checks on event-locked trials: say *why* a trial is rejected instead of dropping it silently."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .force import plateau_window
from .quality import flat_fraction, saturated_fraction
from .signals import Signal
from .windowing import validate_events

REASONS = {
    "outside_recording": "lies outside the recording",
    "too_short": "is too short after trimming",
    "nonfinite": "contains NaN or infinite samples",
    "flat": "has a channel with flat (constant) stretches",
    "saturated": "has clipped (saturated) samples",
    "no_plateau": "has no force plateau",
}


def trial_samples(sig: Signal, onset: float, offset: float, trim_s: float = 0.0) -> tuple[int, int]:
    """Sample range ``[start, stop)`` of ``[onset + trim_s, offset - trim_s]`` in ``sig``."""
    start = round((onset + trim_s - sig.t0) * sig.fs)
    stop = round((offset - trim_s - sig.t0) * sig.fs)
    return start, stop


def geometry_problem(sig: Signal, start: int, stop: int, min_samples: int = 2) -> str | None:
    """``"outside_recording"``, ``"too_short"`` or ``None`` for a sample range."""
    if start < 0 or stop > sig.n_samples:
        return "outside_recording"
    if stop - start < min_samples:
        return "too_short"
    return None


def check_trials(
    sig: Signal,
    events: pd.DataFrame,
    *,
    trim_s: float = 0.0,
    min_duration_s: float | None = None,
    quality: bool = True,
    min_hits: int = 5,
    flat_run: int = 5,
    max_saturated: float = 0.001,
    max_flat: float = 0.01,
) -> pd.DataFrame:
    """One row per trial: is it usable, and if not, why.

    A trial is the stretch ``[onset + trim_s, offset - trim_s]`` of every event, exactly as
    :func:`biosigkit.tables.trial_feature_table` cuts it. Rows follow ``validate_events(events)``
    (sorted by onset), and ``trial`` is that row number. Columns: ``trial, onset, offset, label``,
    the extra columns of ``events`` (e.g. ``level``), then

    ``valid``
        True when no check failed.
    ``reason``
        Codes of the failed checks joined by ``;`` (empty when valid): ``outside_recording``,
        ``too_short`` (fewer than ``min_duration_s`` seconds, default 2 samples), and, with
        ``quality=True``, ``nonfinite``, ``flat`` and ``saturated``. If the events table has a
        boolean ``plateau`` column (see :func:`plateau_events`), ``no_plateau`` is added for the
        trials where it is False.
    ``detail``
        The same, in words, naming the channels at fault.

    The quality checks need the whole recording only for the clipping rails (the extreme values
    of each channel); run them on raw data, before filtering. ``max_saturated`` and ``max_flat``
    are fractions of the trial's samples.
    """
    ev = validate_events(events)
    extra = [c for c in ev.columns if c not in ("onset", "offset", "label")]
    min_samples = 2 if min_duration_s is None else max(2, round(min_duration_s * sig.fs))
    with np.errstate(invalid="ignore"):
        rails_lo = np.nanmin(sig.data, axis=1)
        rails_hi = np.nanmax(sig.data, axis=1)
    rows = []
    for i, rec in enumerate(ev.to_dict("records")):
        codes: list[str] = []
        notes: list[str] = []
        start, stop = trial_samples(sig, rec["onset"], rec["offset"], trim_s)
        problem = geometry_problem(sig, start, stop, min_samples)
        if problem:
            codes.append(problem)
            notes.append(REASONS[problem])
        elif quality:
            window = sig.data[:, start:stop]
            bad_nonfinite = [
                c for c, x in zip(sig.channels, window, strict=True) if not np.isfinite(x).all()
            ]
            bad_flat = [
                c
                for c, x in zip(sig.channels, window, strict=True)
                if flat_fraction(x[np.isfinite(x)], flat_run) > max_flat
            ]
            bad_sat = [
                c
                for c, x, lo, hi in zip(sig.channels, window, rails_lo, rails_hi, strict=True)
                if saturated_fraction(x, lo, hi, min_hits) > max_saturated
            ]
            for code, channels in (
                ("nonfinite", bad_nonfinite),
                ("flat", bad_flat),
                ("saturated", bad_sat),
            ):
                if channels:
                    codes.append(code)
                    notes.append(f"{', '.join(channels)}: {REASONS[code]}")
        if "plateau" in rec and not bool(rec["plateau"]):
            codes.append("no_plateau")
            notes.append(REASONS["no_plateau"])
        rows.append(
            {
                "trial": i,
                "onset": rec["onset"],
                "offset": rec["offset"],
                "label": rec["label"],
                **{k: rec[k] for k in extra},
                "valid": not codes,
                "reason": ";".join(codes),
                "detail": "; ".join(notes),
            }
        )
    columns = ["trial", "onset", "offset", "label", *extra, "valid", "reason", "detail"]
    return pd.DataFrame(rows, columns=columns)


def drop_invalid(events: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    """The events whose trials passed :func:`check_trials`, sorted by onset.

    A ``trial`` column keeps the original trial numbers, so a feature table built from the result
    still lines up with ``report`` (``trial_feature_table`` carries the column through).
    """
    ev = validate_events(events)
    if len(ev) != len(report):
        raise ValueError("report does not belong to this events table (different number of rows)")
    keep = report["valid"].to_numpy(dtype=bool)
    return ev.assign(trial=report["trial"].to_numpy()).loc[keep].reset_index(drop=True)


def plateau_events(
    events: pd.DataFrame,
    force: Signal,
    *,
    channel: str | None = None,
    frac: float = 0.8,
    smooth_s: float = 0.2,
    min_duration_s: float = 0.5,
) -> pd.DataFrame:
    """Narrow every event to the hold phase of its force trace.

    A fixed ``trim_s`` throws away the same time at the start of every trial, however long a
    subject takes to reach the target. This finds the hold instead, with
    :func:`biosigkit.force.plateau_window`, and returns a copy of ``events`` whose ``onset`` and
    ``offset`` are the plateau (the original window is kept in ``onset_trial`` / ``offset_trial``).
    ``plateau`` is True where one was found; where it was not, the original window is kept and
    :func:`check_trials` reports the trial as ``no_plateau``.

    ``force`` is a :class:`~biosigkit.signals.Signal` (its first channel, or ``channel``) on the
    same time axis as the events.
    """
    ev = validate_events(events)
    trace = force.data[0] if channel is None else force.pick([channel]).data[0]
    out = ev.copy()
    out["onset_trial"] = ev["onset"]
    out["offset_trial"] = ev["offset"]
    found = np.zeros(len(ev), dtype=bool)
    for i, (onset, offset) in enumerate(zip(ev["onset"], ev["offset"], strict=True)):
        a = max(0, round((onset - force.t0) * force.fs))
        b = min(force.n_samples, round((offset - force.t0) * force.fs))
        if b - a < 4:
            continue
        window = plateau_window(
            trace[a:b], force.fs, frac=frac, smooth_s=smooth_s, min_duration_s=min_duration_s
        )
        if window is None:
            continue
        out.loc[i, "onset"] = force.t0 + a / force.fs + window[0]
        out.loc[i, "offset"] = force.t0 + a / force.fs + window[1]
        found[i] = True
    out["plateau"] = found
    return out
