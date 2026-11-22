"""Reading and writing signals and event tables."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .signals import Signal
from .windowing import validate_events


def read_csv(
    path: str | Path,
    *,
    fs: float | None = None,
    time_col: str | None = None,
    channels: Sequence[str] | None = None,
    meta: dict[str, Any] | None = None,
    **read_csv_kwargs: Any,
) -> Signal:
    """Load a CSV with one column per channel (and optionally a time column)."""
    df = pd.read_csv(path, **read_csv_kwargs)
    return Signal.from_frame(df, fs=fs, time_col=time_col, channels=channels, meta=meta)


def write_csv(sig: Signal, path: str | Path, *, time_col: str = "time") -> None:
    """Write a signal as ``time, <channel>...`` CSV."""
    sig.to_frame(time_col=time_col).to_csv(path, index=False)


def read_events(path: str | Path) -> pd.DataFrame:
    """Load and validate an events CSV with columns ``onset, offset, label`` (seconds)."""
    return validate_events(pd.read_csv(path))
