"""Shared fixtures and helpers."""

from __future__ import annotations

import numpy as np
import pytest

from biosigkit import Signal


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


def sine(freq: float, fs: float, duration: float, amp: float = 1.0, phase: float = 0.0):
    t = np.arange(round(fs * duration)) / fs
    return amp * np.sin(2 * np.pi * freq * t + phase)


def amplitude_at(x: np.ndarray, fs: float, freq: float) -> float:
    """Amplitude of the spectral component at ``freq`` (exact for whole numbers of cycles)."""
    t = np.arange(x.shape[-1]) / fs
    return float(2 * np.abs(np.mean(x * np.exp(-2j * np.pi * freq * t))))


@pytest.fixture
def make_signal():
    def _make(data, fs=1000.0, **kwargs):
        return Signal(np.asarray(data, dtype=float), fs, **kwargs)

    return _make
