"""Synthetic data generators for demos and tests.

Everything here is simulated: no real participant data ships with the package. The models are
deliberately simple (amplitude-modulated band-limited noise) but capture the properties that
matter for pipelines — inter-subject gain differences, noise floors and mains interference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import signal as sps

from .signals import Signal


def bandlimited_noise(
    n_samples: int,
    fs: float,
    band: tuple[float, float],
    rng: np.random.Generator,
    n_channels: int = 1,
) -> np.ndarray:
    """Unit-variance Gaussian noise band-limited to ``band`` (shape ``(n_channels, n_samples)``)."""
    high = min(band[1], 0.98 * fs / 2)
    sos = sps.butter(4, [band[0], high], btype="bandpass", fs=fs, output="sos")
    x = sps.sosfiltfilt(sos, rng.standard_normal((n_channels, n_samples)), axis=-1)
    return x / x.std(axis=-1, keepdims=True)


def pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance 1/f noise (a rough model of the EEG background spectrum)."""
    spectrum = np.fft.rfft(rng.standard_normal(n_samples))
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1]
    x = np.fft.irfft(spectrum / np.sqrt(freqs), n_samples)
    return x / x.std()


def force_profile(
    duration_s: float,
    fs: float,
    trials: Sequence[tuple[float, float, float]],
    ramp_s: float = 0.5,
) -> np.ndarray:
    """Trapezoidal profile: ``trials`` are ``(onset_s, offset_s, amplitude)`` tuples."""
    xs, ys = [0.0], [0.0]
    for onset, offset, amplitude in sorted(trials):
        ramp = min(ramp_s, (offset - onset) / 2)
        xs += [onset, onset + ramp, offset - ramp, offset]
        ys += [0.0, amplitude, amplitude, 0.0]
    xs.append(duration_s)
    ys.append(0.0)
    t = np.arange(round(duration_s * fs)) / fs
    return np.interp(t, xs, ys)


def emg_from_force(
    force_fraction: np.ndarray,
    fs_force: float,
    fs_emg: float,
    *,
    channel_gains: Sequence[float] | np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
    band: tuple[float, float] = (20.0, 450.0),
    line_amp: float = 0.0,
    line_hz: float = 60.0,
) -> np.ndarray:
    """Simulate surface EMG whose amplitude follows a force trace.

    ``emg_c(t) = gain_c * force(t) * carrier_c(t) + noise floor (+ mains interference)`` where the
    carrier is band-limited Gaussian noise. ``force_fraction`` is force as a fraction of MVC, so
    ``gain`` is the EMG standard deviation at 100 % MVC. Returns ``(n_channels, n_samples)``.
    """
    n_emg = round(len(force_fraction) / fs_force * fs_emg)
    t = np.arange(n_emg) / fs_emg
    env = np.interp(
        t, np.arange(len(force_fraction)) / fs_force, np.clip(force_fraction, 0.0, None)
    )
    gains = np.asarray(channel_gains, dtype=float)
    carrier = bandlimited_noise(n_emg, fs_emg, band, rng, n_channels=len(gains))
    floor = bandlimited_noise(n_emg, fs_emg, (5.0, 0.45 * fs_emg), rng, len(gains)) * noise_std
    emg = gains[:, None] * env[None, :] * carrier + floor
    if line_amp > 0:
        phase = rng.uniform(0.0, 2 * np.pi, size=(len(gains), 2))
        mains = np.sin(2 * np.pi * line_hz * t[None, :] + phase[:, [0]])
        mains += 0.5 * np.sin(2 * np.pi * 2 * line_hz * t[None, :] + phase[:, [1]])
        emg = emg + line_amp * mains
    return emg


@dataclass(eq=False)
class Session:
    """One simulated recording session of a force-matching experiment."""

    subject: str
    emg: Signal
    force: Signal
    target: Signal
    events: pd.DataFrame
    params: dict[str, float] = field(default_factory=dict)


def make_session(
    subject: str,
    *,
    levels: Sequence[int] = (10, 30, 50, 70),
    trials_per_level: int = 6,
    fs_emg: float = 1000.0,
    fs_force: float = 100.0,
    trial_s: float = 6.0,
    rest_s: float = 3.0,
    lead_in_s: float = 3.0,
    ramp_s: float = 0.5,
    trial_gain_sd: float = 0.15,
    gain_drift_sd: float = 0.03,
    line_noise: bool = True,
    seed: int = 0,
) -> Session:
    """Simulate EMG + force for one subject performing randomised %MVC force-matching trials.

    Subject-specific parameters (EMG gain, noise floor, MVC, force steadiness, mains
    interference) are drawn from the seed, producing the strong inter-subject variability that
    makes naive cross-validation optimistic. Within a subject, trials of the same force level
    still differ: ``trial_gain_sd`` is the log-normal spread of a per-trial EMG gain (electrode,
    skin and effort variability), ``gain_drift_sd`` the step of a slow random-walk drift of that
    gain across the session, and the produced force deviates from the target with
    signal-dependent noise. Classification is therefore not trivially perfect.
    """
    rng = np.random.default_rng(seed)
    gain = float(rng.lognormal(mean=np.log(0.8), sigma=0.45))  # mV of EMG s.d. at 100 % MVC
    channel_factor = np.array([1.0, rng.uniform(0.4, 0.8)])
    noise_std = float(rng.uniform(0.008, 0.025))  # mV
    mvc_kgf = float(max(15.0, rng.normal(35.0, 6.0)))
    tracking_cv = float(rng.uniform(0.04, 0.10))  # force error relative to the target
    line_amp = float(rng.uniform(0.02, 0.06)) if line_noise else 0.0

    order = rng.permutation(np.repeat(np.asarray(levels), trials_per_level))
    onsets = lead_in_s + np.arange(len(order)) * (trial_s + rest_s)
    offsets = onsets + trial_s
    duration = float(lead_in_s + len(order) * (trial_s + rest_s))

    target_fraction = force_profile(
        duration,
        fs_force,
        [(on, off, lvl / 100.0) for on, off, lvl in zip(onsets, offsets, order, strict=True)],
        ramp_s,
    )
    sos = sps.butter(2, 0.5, btype="low", fs=fs_force, output="sos")
    wobble = sps.sosfiltfilt(sos, rng.standard_normal(len(target_fraction)))
    wobble /= wobble.std()
    amplitude = 0.01 + tracking_cv * target_fraction  # tremor floor + signal-dependent error
    actual_fraction = np.clip(
        target_fraction + (target_fraction > 0.02) * amplitude * wobble, 0, None
    )

    n_trials = len(order)
    log_gain = np.cumsum(rng.normal(0.0, gain_drift_sd, n_trials)) + rng.normal(
        0.0, trial_gain_sd, n_trials
    )
    t_force = np.arange(len(target_fraction)) / fs_force
    gain_track = np.ones(len(target_fraction))
    for onset, trial_gain in zip(onsets, np.exp(log_gain), strict=True):
        gain_track[t_force >= onset] = trial_gain  # force is ~0 at trial edges: no discontinuity

    emg = emg_from_force(
        actual_fraction * gain_track,
        fs_force,
        fs_emg,
        channel_gains=gain * channel_factor,
        noise_std=noise_std,
        rng=rng,
        line_amp=line_amp,
    )
    events = pd.DataFrame(
        {
            "onset": onsets,
            "offset": offsets,
            "label": [f"{lvl}%" for lvl in order],
            "level": order.astype(int),
        }
    )
    meta = {"subject": subject}
    return Session(
        subject=subject,
        emg=Signal(emg, fs_emg, ("emg1", "emg2"), meta={**meta, "units": "mV"}),
        force=Signal(
            actual_fraction * mvc_kgf, fs_force, ("force",), meta={**meta, "units": "kgf"}
        ),
        target=Signal(
            target_fraction * mvc_kgf, fs_force, ("target",), meta={**meta, "units": "kgf"}
        ),
        events=events,
        params={
            "gain_mv": gain,
            "noise_std_mv": noise_std,
            "mvc_kgf": mvc_kgf,
            "tracking_cv": tracking_cv,
            "line_amp_mv": line_amp,
        },
    )


def make_cohort(n_subjects: int = 8, *, seed: int = 0, **session_kwargs: object) -> list[Session]:
    """Simulate several subjects (ids ``S01``, ``S02``, ...) with independent random streams."""
    children = np.random.SeedSequence(seed).spawn(n_subjects)
    return [
        make_session(f"S{i + 1:02d}", seed=int(child.generate_state(1)[0]), **session_kwargs)  # type: ignore[arg-type]
        for i, child in enumerate(children)
    ]


def make_eeg(
    *,
    n_trials: int = 20,
    trial_s: float = 3.0,
    fs: float = 250.0,
    channels: Sequence[str] = ("C3", "Cz", "C4"),
    erd_depth: Sequence[float] | float = (0.6, 0.4, 0.2),
    alpha_hz: float = 10.5,
    alpha_uv: float = 8.0,
    noise_uv: float = 8.0,
    common_uv: float = 4.0,
    line_uv: float = 0.0,
    seed: int = 0,
) -> tuple[Signal, pd.DataFrame]:
    """Simulate EEG with alternating rest / task blocks and task-related alpha ERD.

    The background is 1/f noise (independent per channel) plus a component shared by all
    channels (what a common-average reference removes). Alpha amplitude drops by ``erd_depth``
    (a fraction, per channel) during task blocks. Returns ``(signal, events)`` where ``events``
    has ``onset, offset, label`` with labels ``rest`` / ``task``.
    """
    rng = np.random.default_rng(seed)
    n_ch = len(channels)
    depth = np.broadcast_to(np.asarray(erd_depth, dtype=float), (n_ch,))
    n = round(2 * n_trials * trial_s * fs)
    t = np.arange(n) / fs
    task = ((t // trial_s) % 2 == 1).astype(float)
    sos = sps.butter(2, 1.0, btype="low", fs=fs, output="sos")
    task = np.clip(sps.sosfiltfilt(sos, task), 0.0, 1.0)  # smooth block transitions
    phase = rng.uniform(0.0, 2 * np.pi, n_ch)
    alpha = (
        alpha_uv
        * (1.0 - depth[:, None] * task[None, :])
        * np.sin(2 * np.pi * alpha_hz * t[None, :] + phase[:, None])
    )
    background = np.stack([pink_noise(n, rng) for _ in range(n_ch)]) * noise_uv
    data = alpha + background + common_uv * pink_noise(n, rng)[None, :]
    if line_uv > 0:
        data = data + line_uv * np.sin(2 * np.pi * 60.0 * t)[None, :]
    blocks = np.arange(2 * n_trials) * trial_s
    events = pd.DataFrame(
        {
            "onset": blocks,
            "offset": blocks + trial_s,
            "label": ["rest" if i % 2 == 0 else "task" for i in range(2 * n_trials)],
        }
    )
    return Signal(data, fs, tuple(channels), meta={"units": "uV"}), events
