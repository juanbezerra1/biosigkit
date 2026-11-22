"""Command-line interface: ``biosigkit demo | features | qc | duplicates | evaluate | sql``."""

from __future__ import annotations

import sqlite3
from functools import partial
from pathlib import Path
from typing import Annotated, NoReturn

import pandas as pd
import typer

from . import __version__
from .demo import run_demo
from .duplicates import find_duplicates
from .eeg import eeg_features
from .emg import emg_features
from .io import read_csv, read_events
from .ml import MODEL_NAMES, cross_validate, within_group_zscore
from .preprocess import EEG_PRESET, EMG_PRESET, preprocess
from .quality import signal_quality
from .storage import list_examples, query, read_example
from .tables import (
    RESERVED_COLUMNS,
    infer_feature_columns,
    trial_feature_table,
    window_feature_table,
)
from .trials import check_trials, drop_invalid

app = typer.Typer(
    help="EMG / EEG / force biosignal toolkit: features, leakage-safe evaluation, SQL.",
    no_args_is_help=True,
    add_completion=False,
)
sql_app = typer.Typer(help="Bundled SQL examples and ad-hoc queries.", no_args_is_help=True)
app.add_typer(sql_app, name="sql")


def _fail(message: str) -> NoReturn:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def _show(df: pd.DataFrame, max_rows: int = 40) -> None:
    typer.echo(df.to_string(index=False, max_rows=max_rows, float_format=lambda v: f"{v:.4g}"))


def _version(value: bool) -> None:
    if value:
        typer.echo(f"biosigkit {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """biosigkit command-line interface."""


@app.command()
def demo(
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")] = Path(
        "biosigkit_demo"
    ),
    seed: Annotated[int, typer.Option(help="Random seed of the simulation.")] = 0,
    subjects: Annotated[int, typer.Option(help="Number of simulated subjects.")] = 8,
    trials_per_level: Annotated[int, typer.Option(help="Trials per force level.")] = 6,
) -> None:
    """Run the full pipeline on synthetic data and write report, figures, CSVs and a database."""
    if subjects < 2:
        _fail("--subjects must be at least 2 (leave-one-subject-out needs two subjects)")
    run_demo(out, seed=seed, n_subjects=subjects, trials_per_level=trials_per_level, log=typer.echo)


@app.command()
def features(
    signal_csv: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="CSV, one column per channel.")
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Feature table CSV.")] = Path(
        "features.csv"
    ),
    fs: Annotated[float | None, typer.Option(help="Sampling rate (Hz); or use --time-col.")] = None,
    time_col: Annotated[str | None, typer.Option(help="Time column (seconds) to infer fs.")] = None,
    events: Annotated[
        Path | None, typer.Option(help="Events CSV with onset,offset,label (seconds).")
    ] = None,
    mode: Annotated[
        str, typer.Option(help="'window' (sliding) or 'trial' (one row/event).")
    ] = "window",
    kind: Annotated[str, typer.Option(help="Signal type: 'emg' or 'eeg'.")] = "emg",
    preprocessing: Annotated[
        bool, typer.Option("--preprocess/--no-preprocess", help="Run the preset chain first.")
    ] = True,
    win: Annotated[float, typer.Option(help="Window length (s) for mode=window.")] = 0.2,
    step: Annotated[float, typer.Option(help="Window step (s) for mode=window.")] = 0.1,
    trim: Annotated[float, typer.Option(help="Seconds trimmed at both ends in mode=trial.")] = 0.0,
    subject: Annotated[str, typer.Option(help="Subject id written to the table.")] = "S01",
    session: Annotated[str, typer.Option(help="Session id written to the table.")] = "S1",
    on_invalid: Annotated[
        str, typer.Option(help="mode=trial: 'raise' (stop at a bad trial) or 'skip' it.")
    ] = "raise",
    check_quality: Annotated[
        bool,
        typer.Option(
            "--check-quality/--no-check-quality",
            help="mode=trial: also reject trials with NaN, flat or clipped samples (raw data).",
        ),
    ] = False,
    rejected_out: Annotated[
        Path | None, typer.Option(help="mode=trial: write the rejected trials and why to this CSV.")
    ] = None,
) -> None:
    """Preprocess a signal CSV and write a window- or trial-level feature table."""
    if mode not in {"window", "trial"}:
        _fail("--mode must be 'window' or 'trial'")
    if kind not in {"emg", "eeg"}:
        _fail("--kind must be 'emg' or 'eeg'")
    if on_invalid not in {"raise", "skip"}:
        _fail("--on-invalid must be 'raise' or 'skip'")
    if mode == "trial" and events is None:
        _fail("--mode trial needs --events")
    if mode == "window" and (check_quality or rejected_out is not None):
        _fail("--check-quality and --rejected-out only apply to --mode trial")
    rejected = None
    try:
        raw = read_csv(signal_csv, fs=fs, time_col=time_col)
        sig = preprocess(raw, EMG_PRESET if kind == "emg" else EEG_PRESET) if preprocessing else raw
        feature_fn = emg_features if kind == "emg" else partial(eeg_features, nperseg_s=1.0)
        ev = read_events(events) if events is not None else None
        meta = {"subject": subject, "session": session}
        if mode == "window":
            table = window_feature_table(
                sig, win_s=win, step_s=step, feature_fn=feature_fn, events=ev, meta=meta
            )
        else:
            assert ev is not None
            # the quality checks need the raw samples: filtering smooths clipping away
            report = check_trials(raw, ev, trim_s=trim, quality=check_quality)
            rejected = report.loc[~report["valid"]].drop(columns="valid")
            if len(rejected) and on_invalid == "raise":
                first = rejected.iloc[0]
                _fail(
                    f"{len(rejected)} of {len(report)} trial(s) rejected; the first, trial "
                    f"{first['trial']} ({first['onset']}-{first['offset']} s), "
                    f"{first['detail']}. Use --on-invalid skip to leave them out."
                )
            table = trial_feature_table(
                sig, drop_invalid(ev, report), feature_fn=feature_fn, trim_s=trim, meta=meta
            )
    except (ValueError, KeyError, OSError) as exc:
        _fail(str(exc))
    table.to_csv(output, index=False)
    typer.echo(f"Wrote {len(table)} rows x {table.shape[1]} columns to {output}")
    if rejected is not None:
        if rejected_out is not None:
            rejected.to_csv(rejected_out, index=False)
        if len(rejected):
            reasons = rejected["reason"].str.split(";").explode().value_counts()
            summary = ", ".join(f"{n} {reason}" for reason, n in reasons.items())
            where = f"; see {rejected_out}" if rejected_out is not None else ""
            typer.echo(f"Skipped {len(rejected)} trial(s): {summary}{where}")


@app.command()
def qc(
    signal_csv: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="CSV, one column per channel.")
    ],
    fs: Annotated[float | None, typer.Option(help="Sampling rate (Hz); or use --time-col.")] = None,
    time_col: Annotated[str | None, typer.Option(help="Time column (seconds) to infer fs.")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Also write a CSV.")] = None,
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit with status 1 if any channel has an issue.")
    ] = False,
) -> None:
    """Signal-quality report per channel: clipping, flat lines, NaN and mains interference."""
    try:
        sig = read_csv(signal_csv, fs=fs, time_col=time_col)
        report = signal_quality(sig)
    except (ValueError, KeyError, OSError) as exc:
        _fail(str(exc))
    _show(report.drop(columns=["n_samples", "issues"]))
    for row in report.loc[~report["ok"]].itertuples():
        typer.echo(f"  {row.channel}: {row.issues}")
    if output is not None:
        report.to_csv(output, index=False)
    bad = int((~report["ok"]).sum())
    typer.echo(f"{len(report) - bad} of {len(report)} channel(s) ok.")
    if strict and bad:
        raise typer.Exit(1)


@app.command()
def duplicates(
    files: Annotated[
        list[Path], typer.Argument(exists=True, dir_okay=False, help="Signal CSVs to compare.")
    ],
    fs: Annotated[float | None, typer.Option(help="Sampling rate (Hz); or use --time-col.")] = None,
    time_col: Annotated[str | None, typer.Option(help="Time column (seconds) to infer fs.")] = None,
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit with status 1 if a duplicate is found.")
    ] = False,
) -> None:
    """Find recordings (or channels) that hold exactly the same samples under different names."""
    try:
        recordings = {str(f): read_csv(f, fs=fs, time_col=time_col) for f in files}
        report = find_duplicates(recordings)
    except (ValueError, KeyError, OSError) as exc:
        _fail(str(exc))
    if report.empty:
        typer.echo(f"No duplicates among {len(recordings)} recording(s).")
        return
    _show(report)
    if strict:
        raise typer.Exit(1)


@app.command()
def evaluate(
    features_csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    label: Annotated[str, typer.Option(help="Label column.")] = "label",
    group: Annotated[str, typer.Option(help="Group (subject) column.")] = "subject",
    model: Annotated[str, typer.Option(help=f"One of {', '.join(MODEL_NAMES)}.")] = "lda",
    scheme: Annotated[str, typer.Option(help="loso | group_kfold | kfold (leaky).")] = "loso",
    normalize: Annotated[
        bool, typer.Option("--normalize", help="Z-score features within each group first.")
    ] = False,
) -> None:
    """Cross-validate a classifier on a feature table without subject leakage."""
    try:
        df = pd.read_csv(features_csv)
        cols = infer_feature_columns(df, RESERVED_COLUMNS | {label, group})
        if normalize:
            df = within_group_zscore(df, cols, group)
        result = cross_validate(df, cols, label, group, model=model, scheme=scheme)  # type: ignore[arg-type]
    except (ValueError, KeyError, AssertionError) as exc:
        _fail(str(exc))
    typer.echo(f"model={model} scheme={scheme} normalize={normalize} features={len(cols)}")
    for key, value in result.summary().items():
        typer.echo(f"  {key:>20}: {value:.3f}")
    typer.echo("\nConfusion matrix (rows = true label):")
    _show(result.confusion().reset_index())


@sql_app.command("list")
def sql_list() -> None:
    """List the bundled example queries."""
    for name in list_examples():
        typer.echo(name)


@sql_app.command("show")
def sql_show(name: Annotated[str, typer.Argument(help="Example name, see `sql list`.")]) -> None:
    """Print the text of an example query."""
    try:
        typer.echo(read_example(name))
    except KeyError as exc:
        _fail(str(exc.args[0]))


@sql_app.command("run")
def sql_run(
    db: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="SQLite database.")],
    example: Annotated[str | None, typer.Option("--example", "-e", help="Bundled example.")] = None,
    sql: Annotated[str | None, typer.Option("--query", "-q", help="Your own SELECT.")] = None,
) -> None:
    """Run a bundled example or your own SELECT against a SQLite database (read-only)."""
    if (example is None) == (sql is None):
        _fail("give exactly one of --example or --query")
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA query_only = ON")  # this command only ever reads
    try:
        _show(query(conn, read_example(example) if example is not None else str(sql)))
    except (KeyError, sqlite3.Error, pd.errors.DatabaseError) as exc:
        _fail(str(exc.args[0]) if isinstance(exc, KeyError) else str(exc))
    finally:
        conn.close()
