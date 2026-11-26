"""End-to-end demo: synthetic cohort -> preprocessing -> features -> SQLite -> honest evaluation.

Run it with ``biosigkit demo``. Everything is simulated (see :mod:`biosigkit.synthetic`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .duplicates import find_duplicates, signal_hash
from .eeg import BANDS, band_power, erd_percent, psd, reject_by_ptp
from .emg import baseline_threshold, detect_activations, emg_features, linear_envelope
from .force import steadiness_cv, tracking_error
from .ml import compare_scenarios, cross_validate, within_group_zscore
from .preprocess import EEG_PRESET, EMG_PRESET, preprocess
from .quality import signal_quality
from .storage import (
    connect,
    list_examples,
    load_feature_table,
    query,
    read_example,
    register_recording,
    save_feature_table,
)
from .synthetic import Session, make_cohort, make_eeg
from .tables import infer_feature_columns, summarize, trial_feature_table, window_feature_table
from .trials import check_trials, plateau_events
from .viz import plot_confusion, plot_eeg_erd, plot_emg_overview, plot_scenarios, save_figure
from .windowing import make_epochs

TRIAL_FEATURES = ("mav", "rms", "wl", "zc", "ssc", "mnf", "mdf")
TRIM_S = 1.0  # drop the force ramps at both ends of every trial


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def _tracking_rows(session: Session) -> list[dict[str, Any]]:
    rows = []
    for ev in session.events.itertuples():
        lo, hi = ev.onset + TRIM_S, ev.offset - TRIM_S
        force = session.force.crop(lo, hi).data[0]
        target = session.target.crop(lo, hi).data[0]
        err = tracking_error(force, target)
        rows.append(
            {
                "subject": session.subject,
                "level": int(ev.level),
                "target_kgf": float(target.mean()),
                "rmse_kgf": err["rmse"],
                "rmse_pct_of_target": 100 * err["rmse"] / float(target.mean()),
                "bias_kgf": err["bias"],
                "cv": steadiness_cv(force),
            }
        )
    return rows


def _detection_stats(events: pd.DataFrame, acts: pd.DataFrame) -> dict[str, Any]:
    hits, latencies = 0, []
    for ev in events.itertuples():
        overlap = acts[(acts["onset"] < ev.offset) & (acts["offset"] > ev.onset)]
        if len(overlap):
            hits += 1
            latencies.append(float(overlap["onset"].iloc[0] - ev.onset))
    false_positives = sum(
        not ((events["onset"] < a.offset) & (events["offset"] > a.onset)).any()
        for a in acts.itertuples()
    )
    return {
        "trials": len(events),
        "detected": hits,
        "false_positives": int(false_positives),
        "latencies": latencies,
    }


def _data_checks(
    cohort: list[Session], qc: pd.DataFrame, trial_checks: pd.DataFrame
) -> dict[str, Any]:
    """Numbers for the "Data checks" section: quality, trial validity, force plateau, duplicates."""
    held, fixed = [], []
    n_found = n_trials = 0
    for s in cohort:
        plateau = plateau_events(s.events, s.force)
        n_trials += len(plateau)
        n_found += int(plateau["plateau"].sum())
        held += (plateau["offset"] - plateau["onset"])[plateau["plateau"]].tolist()
        fixed += (s.events["offset"] - s.events["onset"] - 2 * TRIM_S).tolist()
    copies = find_duplicates({s.subject: s.emg for s in cohort})
    return {
        "channels_checked": len(qc),
        "channels_ok": int(qc["ok"].sum()),
        "max_mains_fraction": float(qc["mains_fraction"].max()),
        "max_saturated_fraction": float(qc["saturated_fraction"].max()),
        "trials_checked": len(trial_checks),
        "trials_valid": int(trial_checks["valid"].sum()),
        "plateau_found": n_found,
        "plateau_trials": n_trials,
        "median_hold_s": float(np.median(held)) if held else float("nan"),
        "median_fixed_trim_window_s": float(np.median(fixed)) if fixed else float("nan"),
        "duplicate_groups": len(copies),
    }


def _eeg_demo(out: Path, seed: int) -> dict[str, Any]:
    sig, events = make_eeg(n_trials=20, seed=seed + 1, line_uv=3.0)
    clean = preprocess(sig, EEG_PRESET)
    epochs = {}
    for label in ("rest", "task"):
        ep = make_epochs(clean, events[events["label"] == label], tmin=0.5, tmax=2.5)
        epochs[label] = ep.select(reject_by_ptp(ep.data, 150.0))
    freqs, p_rest = psd(epochs["rest"].data, clean.fs, nperseg_s=1.0)
    _, p_task = psd(epochs["task"].data, clean.fs, nperseg_s=1.0)
    alpha = {
        k: band_power(v.data, clean.fs, BANDS["alpha"], nperseg_s=1.0).mean(axis=0)
        for k, v in epochs.items()
    }
    erd = erd_percent(alpha["task"], alpha["rest"])
    erd_by_channel = {ch: float(v) for ch, v in zip(clean.channels, erd, strict=True)}
    ci = clean.channels.index("C3")
    fig = plot_eeg_erd(
        freqs, p_rest.mean(axis=0)[ci], p_task.mean(axis=0)[ci], erd_by_channel, channel="C3"
    )
    save_figure(fig, out / "figures" / "eeg_erd.png")
    pd.DataFrame(
        {
            "channel": list(clean.channels),
            "alpha_power_rest": alpha["rest"],
            "alpha_power_task": alpha["task"],
            "erd_percent": erd,
        }
    ).to_csv(out / "eeg_alpha_erd.csv", index=False)
    return {
        "mains_detected_hz": clean.meta["preprocess"]["notch_applied_hz"],
        "epochs_kept": {k: len(v) for k, v in epochs.items()},
        "alpha_erd_percent": erd_by_channel,
    }


def _markdown_table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    def cell(value: Any) -> str:
        return fmt.format(value) if isinstance(value, float) else str(value)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def _report(summary: dict[str, Any], results: pd.DataFrame, tracking: pd.DataFrame) -> str:
    det = summary["detection"]
    chk = summary["checks"]
    shown = results[
        ["model", "scheme", "normalization", "accuracy", "balanced_accuracy", "f1_macro"]
    ]
    lines = [
        "# biosigkit demo report",
        "",
        "All data below is **synthetic** (see `biosigkit.synthetic`); it illustrates the pipeline, "
        "it is not a scientific result.",
        "",
        f"- Subjects: {summary['n_subjects']}, trials: {summary['n_trials']}, "
        f"windows: {summary['n_windows']}, classes: {summary['n_classes']} "
        f"(chance = {summary['chance']:.0%})",
        f"- Mains interference detected per subject: {summary['mains_detected']}",
        "",
        "## Force-level classification from trial-level EMG features",
        "",
        _markdown_table(shown),
        "",
        "`kfold` ignores subjects (leaky); `loso` holds out one whole subject per fold; "
        "`within_group` z-scores features inside each subject.",
        "",
        "## Force tracking (hold phase)",
        "",
        _markdown_table(tracking),
        "",
        "## Muscle-activation detection (channel emg1, threshold = baseline mean + 3 SD)",
        "",
        f"- Trials with a detection: {det['detected']}/{det['trials']}, "
        f"false positives: {det['false_positives']}, "
        f"median onset latency: {det['median_onset_latency_s']:.2f} s",
        "",
        "## EEG alpha ERD (task vs rest)",
        "",
        *[f"- {ch}: {v:.1f}%" for ch, v in summary["eeg"]["alpha_erd_percent"].items()],
        "",
        "## Data checks (raw EMG, before preprocessing)",
        "",
        f"- Signal quality: {chk['channels_ok']}/{chk['channels_checked']} channels pass "
        f"(largest mains share of power {chk['max_mains_fraction']:.1%}, "
        f"largest clipped share {chk['max_saturated_fraction']:.2%}); per channel in `qc.csv`",
        f"- Trials: {chk['trials_valid']}/{chk['trials_checked']} valid; a rejected trial would "
        "be listed with its reason in `trial_checks.csv`",
        f"- Force plateau found in {chk['plateau_found']}/{chk['plateau_trials']} trials, median "
        f"hold {chk['median_hold_s']:.1f} s (the fixed {TRIM_S:g} s trim above keeps "
        f"{chk['median_fixed_trim_window_s']:.1f} s)",
        f"- Duplicate recordings: {chk['duplicate_groups']} groups among the subjects "
        "(`sql/duplicate_recordings.csv`)",
        "",
        "## SQL",
        "",
        f"- Contraction episodes recovered by the gaps-and-islands query: "
        f"{summary['episodes_from_sql']} (trials simulated: {summary['episodes_expected']})",
        "- Result of every bundled query: `sql/*.csv`; database: `biosigkit_demo.sqlite`",
        "",
    ]
    return "\n".join(lines)


def run_demo(
    out_dir: str | Path,
    *,
    seed: int = 0,
    n_subjects: int = 8,
    trials_per_level: int = 6,
    log: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Run the whole pipeline on a simulated cohort and write results into ``out_dir``.

    Returns a JSON-serialisable summary (also saved as ``summary.json``).
    """
    say = log or (lambda _msg: None)
    out = Path(out_dir)
    for sub in ("figures", "sql"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    say(f"[1/6] Simulating {n_subjects} subjects x {4 * trials_per_level} force-matching trials")
    cohort = make_cohort(n_subjects, seed=seed, trials_per_level=trials_per_level)

    say("[2/6] Preprocessing, window/trial feature tables, force tracking, activation detection")
    db_path = out / "biosigkit_demo.sqlite"
    db_path.unlink(missing_ok=True)
    conn = connect(db_path)
    trial_fn = partial(emg_features, names=TRIAL_FEATURES)
    window_tables, trial_tables, tracking, detection = [], [], [], []
    qc_tables, check_tables = [], []
    mains: dict[str, list[float]] = {}
    overview = None
    for s in cohort:
        emg = preprocess(s.emg, EMG_PRESET)
        info = emg.meta["preprocess"]
        mains[s.subject] = info["notch_applied_hz"]
        register_recording(
            conn, s.subject, "S1", fs=emg.fs, preprocess=info, content_sha256=signal_hash(s.emg)
        )
        qc_tables.append(signal_quality(s.emg, meta={"subject": s.subject}))
        check_tables.append(check_trials(s.emg, s.events, trim_s=TRIM_S).assign(subject=s.subject))
        meta = {"subject": s.subject, "session": "S1"}
        window_tables.append(
            window_feature_table(emg, win_s=0.2, step_s=0.1, events=s.events, meta=meta)
        )
        trial_tables.append(
            trial_feature_table(emg, s.events, feature_fn=trial_fn, trim_s=TRIM_S, meta=meta)
        )
        tracking.extend(_tracking_rows(s))
        env = linear_envelope(emg.pick(["emg1"]), cutoff=5.0)
        baseline = (0.5, float(s.events["onset"].iloc[0]) - 0.5)  # resting lead-in
        acts = detect_activations(
            env.data[0], env.fs, baseline=baseline, k=3.0, min_duration_s=1.0, min_gap_s=0.3
        )
        detection.append(_detection_stats(s.events, acts))
        if overview is None:
            thr = baseline_threshold(env.data[0], env.fs, baseline, k=3.0)
            overview = (emg, env, acts, s.events, thr)
    qc = pd.concat(qc_tables, ignore_index=True)
    trial_checks = pd.concat(check_tables, ignore_index=True)
    qc.to_csv(out / "qc.csv", index=False)
    trial_checks.to_csv(out / "trial_checks.csv", index=False)
    windows = pd.concat(window_tables, ignore_index=True)
    trials = pd.concat(trial_tables, ignore_index=True)
    n_windows = save_feature_table(conn, windows, kind="window")
    n_trials = save_feature_table(conn, trials, kind="trial")
    say(f"      stored {n_windows} windows and {n_trials} trials in {db_path.name}")

    say("[3/6] Cross-validating classifiers on trial features loaded back from SQL")
    table = load_feature_table(conn, "trial")
    cols = infer_feature_columns(table)
    n_classes = int(table["label"].nunique())
    results = compare_scenarios(table, cols, "label", "subject")
    results.to_csv(out / "cv_results.csv", index=False)
    best = cross_validate(
        within_group_zscore(table, cols, "subject"), cols, "label", "subject", model="lda"
    )
    best.folds.to_csv(out / "loso_folds_lda_zscore.csv", index=False)

    say("[4/6] Running the bundled SQL examples")
    sql_rows = {}
    for name in list_examples():
        df = query(conn, read_example(name))
        df.to_csv(out / "sql" / f"{name}.csv", index=False)
        sql_rows[name] = len(df)
    conn.close()

    say("[5/6] Drawing figures")
    emg, env, acts, events, thr = overview  # type: ignore[misc]
    save_figure(
        plot_emg_overview(emg, env, acts, events, channel="emg1", threshold=thr),
        out / "figures" / "emg_overview.png",
    )
    save_figure(plot_scenarios(results, chance=1 / n_classes), out / "figures" / "cv_scenarios.png")
    save_figure(
        plot_confusion(best.confusion(), title="LDA, LOSO + within-subject z-score"),
        out / "figures" / "confusion_matrix.png",
    )

    say("[6/6] EEG: alpha desynchronisation on synthetic C3/Cz/C4")
    eeg = _eeg_demo(out, seed)

    tracking_df = pd.DataFrame(tracking)
    tracking_df.to_csv(out / "force_tracking_trials.csv", index=False)
    tracking_summary = summarize(
        tracking_df, by=["level"], value_cols=["rmse_kgf", "rmse_pct_of_target"], stats=("mean",)
    )
    latencies = [v for d in detection for v in d["latencies"]]
    summary: dict[str, Any] = {
        "n_subjects": n_subjects,
        "n_trials": int(n_trials),
        "n_windows": int(n_windows),
        "n_classes": n_classes,
        "chance": 1 / n_classes,
        "mains_detected": mains,
        "scenarios": results[
            ["model", "scheme", "normalization", "accuracy", "balanced_accuracy", "f1_macro"]
        ].to_dict("records"),
        "episodes_from_sql": sql_rows["activation_episodes"],
        "episodes_expected": int(n_trials),
        "detection": {
            "trials": sum(d["trials"] for d in detection),
            "detected": sum(d["detected"] for d in detection),
            "false_positives": sum(d["false_positives"] for d in detection),
            "median_onset_latency_s": float(np.median(latencies)) if latencies else float("nan"),
        },
        "checks": _data_checks(cohort, qc, trial_checks),
        "eeg": eeg,
    }
    # explicit UTF-8 and LF: the same bytes on every OS (Windows defaults to cp1252)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8", newline="\n"
    )
    (out / "report.md").write_text(
        _report(summary, results, tracking_summary), encoding="utf-8", newline="\n"
    )
    say("\nValidation scenarios (accuracy over all held-out predictions):")
    say(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    say(f"\nResults written to {out}/ (report.md, figures/, sql/, biosigkit_demo.sqlite)")
    return summary
