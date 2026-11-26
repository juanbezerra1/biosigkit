"""Find recordings (or single channels) that hold exactly the same samples under different names.

The same file copied into two session folders is an easy mistake and a quiet leak:
leave-one-subject-out cannot see that "S07" and "S12" are one person, so a copy of the test
subject sits in the training set. Comparing a hash of the *samples* catches it whatever the file
names or channel names say.
"""

from __future__ import annotations

import hashlib
import struct
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .signals import Signal


def _digest(data: np.ndarray, fs: float) -> str:
    h = hashlib.sha256()
    h.update(struct.pack("<dqq", fs, *(data.shape if data.ndim == 2 else (1, data.shape[0]))))
    h.update(np.ascontiguousarray(data, dtype="<f8").tobytes())
    return h.hexdigest()


def signal_hash(sig: Signal) -> str:
    """SHA-256 (hex) of a signal's samples and sampling rate.

    Channel names, ``t0`` and metadata are ignored on purpose: the same samples under another
    name are still the same recording. Changing one sample, the sampling rate or the order of the
    channels changes the hash.
    """
    return _digest(sig.data, sig.fs)


def channel_hash(sig: Signal, channel: str) -> str:
    """SHA-256 (hex) of one channel's samples and the sampling rate."""
    return _digest(sig.pick([channel]).data[0], sig.fs)


def find_duplicates(recordings: Mapping[str, Signal]) -> pd.DataFrame:
    """Groups of recordings (and of channels) that are exact copies of each other.

    ``recordings`` maps a name (file, ``subject/session``, ...) to its signal. Columns of the
    result: ``level`` (``"recording"`` or ``"channel"``), ``n`` (how many copies), ``hash`` (the
    first 12 characters), and ``members`` (names joined by ``", "``; channels as
    ``name/channel``). Empty when everything is unique.

    Channel-level groups are reported when a channel is repeated *within* a recording or shared
    by recordings that are not copies of each other as a whole (a pair of identical recordings
    would otherwise be reported again once per channel). Channels that never change (an
    unplugged electrode) are skipped: they would match each other without being copies.
    """
    whole: dict[str, list[str]] = defaultdict(list)
    for name, sig in recordings.items():
        whole[signal_hash(sig)].append(name)
    copies = {h: names for h, names in whole.items() if len(names) > 1}
    explained = [set(names) for names in copies.values()]

    chans: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name, sig in recordings.items():
        for channel, x in zip(sig.channels, sig.data, strict=True):
            if np.ptp(x) == 0:
                continue
            chans[_digest(x, sig.fs)].append((name, channel))

    rows = [
        {"level": "recording", "n": len(names), "hash": h[:12], "members": ", ".join(names)}
        for h, names in copies.items()
    ]
    for h, members in chans.items():
        if len(members) < 2:
            continue
        owners = [name for name, _ in members]
        one_each = len(set(owners)) == len(owners)
        if one_each and any(set(owners) <= group for group in explained):
            continue  # already explained by identical recordings
        rows.append(
            {
                "level": "channel",
                "n": len(members),
                "hash": h[:12],
                "members": ", ".join(f"{name}/{channel}" for name, channel in members),
            }
        )
    return pd.DataFrame(rows, columns=["level", "n", "hash", "members"])
