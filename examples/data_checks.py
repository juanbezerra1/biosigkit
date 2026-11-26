"""Check a recording before modelling: quality, rejected trials with reasons, hold phase, copies."""

import numpy as np

from biosigkit import (
    Signal,
    check_trials,
    drop_invalid,
    find_duplicates,
    make_session,
    plateau_events,
    signal_quality,
)

session = make_session("S01", trials_per_level=2, seed=3)
emg = session.emg

# Damage the raw recording the way real ones get damaged: the amplifier clips on channel emg1
# (so the strongest contractions are flattened) and a few samples go missing on emg2.
data = emg.data.copy()
limit = 0.5 * np.abs(data[0]).max()
data[0] = np.clip(data[0], -limit, limit)
start = int((session.events["onset"].iloc[4] + 2.0) * emg.fs)
data[1, start : start + 25] = np.nan
damaged = Signal(data, emg.fs, emg.channels)

# 1. One row per channel: how much of it is unusable, and why.
quality = signal_quality(damaged)
print(quality[["channel", "saturated_fraction", "nonfinite_fraction", "mains_fraction", "ok"]])
print()
print(quality.loc[~quality["ok"], ["channel", "issues"]].to_string(index=False))

# 2. One row per trial: usable or not, and the reason. Nothing is dropped silently.
report = check_trials(damaged, session.events, trim_s=1.0)
print()
print(report.loc[~report["valid"], ["trial", "label", "reason"]].to_string(index=False))
usable = drop_invalid(session.events, report)  # keeps the original trial numbers

# 3. Take features from the hold phase of the force trace instead of trimming a fixed second.
hold = plateau_events(usable, session.force)
print()
held = np.median(hold["offset"] - hold["onset"])
whole = np.median(usable["offset"] - usable["onset"])
found = int(hold["plateau"].sum())
print(f"{len(usable)} usable trials; hold found in {found}, median {held:.1f} s of {whole:.1f} s")

# 4. The same samples under two names are one recording, and a leak for subject-wise validation.
copy = Signal(emg.data.copy(), emg.fs, ("a", "b"))
print()
print(find_duplicates({"S01/S1": emg, "S09/S1": copy}).to_string(index=False))
