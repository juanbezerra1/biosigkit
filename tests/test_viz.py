"""Figures: they build headlessly, save as real PNGs and keep their structural promises."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from biosigkit import make_session
from biosigkit.emg import baseline_threshold, detect_activations, linear_envelope
from biosigkit.viz import (
    BASELINE,
    GRID,
    MUTED,
    PRIMARY,
    SECONDARY,
    SERIES,
    SURFACE,
    _longest_gap_center,
    plot_confusion,
    plot_eeg_erd,
    plot_emg_overview,
    plot_scenarios,
    save_figure,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def scenario_results(models=("lda", "rf"), accuracy=0.7) -> pd.DataFrame:
    scenarios = [("kfold", "none"), ("loso", "none"), ("loso", "within_group")]
    rows = [
        {"model": m, "scheme": sc, "normalization": nm, "accuracy": accuracy - 0.1 * i}
        for m in models
        for i, (sc, nm) in enumerate(scenarios)
    ]
    return pd.DataFrame(rows)


def test_palette_tokens_are_valid_hex_colours():
    for colour in (SURFACE, PRIMARY, SECONDARY, MUTED, GRID, BASELINE, *SERIES):
        assert re.fullmatch(r"#[0-9a-f]{6}", colour)
    assert len(set(SERIES)) == len(SERIES)


def test_emg_overview_has_signal_and_envelope_panels():
    session = make_session("S01", trials_per_level=1, seed=1)
    env = linear_envelope(session.emg.pick(["emg1"]))
    baseline = (0.5, 2.5)
    acts = detect_activations(env.data[0], env.fs, baseline=baseline, min_duration_s=1.0)
    threshold = baseline_threshold(env.data[0], env.fs, baseline)
    fig = plot_emg_overview(
        session.emg, env, acts, session.events, channel="emg1", threshold=threshold, tmax=20.0
    )
    assert isinstance(fig, Figure) and len(fig.axes) == 2
    assert fig.axes[1].get_xlim() == (0.0, 20.0)
    assert any(t.get_text() == "threshold" for t in fig.axes[1].texts)


def test_scenario_chart_has_one_bar_per_model_and_scenario():
    fig = plot_scenarios(scenario_results(), chance=0.25)
    (ax,) = fig.axes
    assert len(ax.patches) == 2 * 3
    assert [t.get_text() for t in ax.get_xticklabels()] == ["LDA", "Random forest"]
    assert any("chance" in t.get_text() for t in ax.texts)
    assert fig.legends and len(fig.legends[0].get_texts()) == 3  # legend: >= 2 series


def test_scenario_chart_refuses_more_scenarios_than_palette_slots():
    extra = scenario_results()
    extra = pd.concat([extra, extra.assign(scheme="group_kfold", normalization="none")])
    with pytest.raises(ValueError, match="at most"):
        plot_scenarios(extra)


def test_confusion_matrix_prints_every_count():
    cm = pd.DataFrame([[5, 1], [2, 4]], index=["10%", "30%"], columns=["10%", "30%"])
    fig = plot_confusion(cm, title="demo")
    ax = fig.axes[0]
    assert sorted(t.get_text() for t in ax.texts) == ["1", "2", "4", "5"]
    assert len(fig.axes) == 2  # heat map + colour bar


def test_eeg_figure_shows_psd_and_signed_erd_bars():
    freqs = np.linspace(0, 50, 101)
    rest = 1.0 + 20.0 * np.exp(-((freqs - 10.0) ** 2) / 2)
    task = 1.0 + 5.0 * np.exp(-((freqs - 10.0) ** 2) / 2)
    fig = plot_eeg_erd(freqs, rest, task, {"C3": -60.0, "C4": -20.0}, channel="C3")
    left, right = fig.axes
    assert len(left.lines) >= 2
    assert [p.get_height() for p in right.patches] == [-60.0, -20.0]
    assert [t.get_text() for t in right.texts] == ["−60%", "−20%"]  # typographic minus


def test_longest_gap_center():
    acts = pd.DataFrame({"onset": [2.0, 10.0], "offset": [4.0, 12.0]})
    # gaps: (0, 2), (4, 10), (12, 20) -> the last one is the longest
    assert _longest_gap_center(acts, 20.0) == 16.0
    assert _longest_gap_center(acts.iloc[0:0], 20.0) == 10.0
    straddling = pd.DataFrame({"onset": [15.0], "offset": [30.0]})  # clipped to tmax
    assert _longest_gap_center(straddling, 20.0) == 7.5


def test_save_figure_writes_a_png_and_creates_parent_folders(tmp_path):
    fig = plot_confusion(pd.DataFrame([[1, 0], [0, 1]], index=["a", "b"], columns=["a", "b"]))
    path = save_figure(fig, tmp_path / "nested" / "dir" / "cm.png", dpi=60)
    assert path.read_bytes().startswith(PNG_MAGIC)
