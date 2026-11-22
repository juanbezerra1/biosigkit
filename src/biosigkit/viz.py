"""Static matplotlib figures for the demo and the README.

Figures are built on :class:`matplotlib.figure.Figure` directly (no pyplot), so they work in
headless environments and leave no global state behind. Styling: solid hairline grid, thin marks,
text in ink colours (never in a series colour), categorical hues assigned in a fixed order, and a
legend whenever there are two or more series.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .signals import Signal

SURFACE = "#fcfcfb"
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# Categorical slots 1-3 (blue, orange, aqua), assigned in this fixed order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
# Sequential blue ramp, steps 100 -> 700.
_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]  # fmt: skip
BLUE_CMAP = LinearSegmentedColormap.from_list("biosigkit_blue", _BLUE_RAMP)

MODEL_LABELS = {
    "lda": "LDA",
    "svm_rbf": "SVM (RBF)",
    "svm_linear": "SVM (linear)",
    "rf": "Random forest",
    "logreg": "Logistic reg.",
}
SCENARIO_LABELS = {
    ("kfold", "none"): "Random split (leaky)",
    ("loso", "none"): "Leave-one-subject-out",
    ("loso", "within_group"): "LOSO + within-subject z-score",
}


def _figure(width: float, height: float) -> Figure:
    return Figure(figsize=(width, height), facecolor=SURFACE, layout="constrained")


def _style_axes(ax, grid_axis: str | None = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY, labelsize=8, length=3, width=0.75)
    ax.grid(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.75)
        ax.set_axisbelow(True)
    ax.xaxis.label.set_color(SECONDARY)
    ax.yaxis.label.set_color(SECONDARY)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=10.5, fontweight="bold", color=PRIMARY)


def _longest_gap_center(activations: pd.DataFrame, tmax: float) -> float:
    """Centre of the longest stretch in ``[0, tmax]`` without a detected activation."""
    edges = [0.0]
    for row in activations.sort_values("onset").itertuples():
        if row.onset >= tmax:
            break
        edges += [max(float(row.onset), 0.0), min(float(row.offset), tmax)]
    edges.append(tmax)
    start, stop = max(zip(edges[0::2], edges[1::2], strict=True), key=lambda g: g[1] - g[0])
    return (start + stop) / 2


def _minus(text: str) -> str:
    return text.replace("-", "−")  # typographic minus, like the axis tick labels


def save_figure(fig: Figure, path: str | Path, dpi: int = 160) -> Path:
    """Save as PNG (or any format matplotlib supports) on the figure surface colour."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    return path


def plot_emg_overview(
    emg: Signal,
    envelope: Signal,
    activations: pd.DataFrame,
    events: pd.DataFrame,
    *,
    channel: str,
    threshold: float,
    tmax: float = 40.0,
) -> Figure:
    """Raw EMG (top) and its linear envelope with the detection threshold and activations."""
    fig = _figure(10, 4.8)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    t, te = emg.times(), envelope.times()
    mask, mask_e = t <= tmax, te <= tmax
    ax1.plot(t[mask], emg.data[emg.channels.index(channel), mask], color=SERIES[0], lw=0.5)
    ax1.set_ylabel("EMG (mV)")
    _style_axes(ax1)
    _title(ax1, f"Preprocessed surface EMG, channel {channel}")

    env = envelope.data[envelope.channels.index(channel), mask_e]
    ax2.plot(te[mask_e], env, color=SERIES[0], lw=1.5)
    for row in activations.itertuples():
        if row.onset < tmax:
            ax2.axvspan(row.onset, min(row.offset, tmax), color=SERIES[1], alpha=0.16, lw=0)
    ax2.axhline(threshold, color=SECONDARY, lw=0.75)
    top = max(float(env.max()), threshold) * 1.18
    ax2.set_ylim(0, top)
    ax2.text(
        _longest_gap_center(activations, tmax), threshold, "threshold",
        ha="center", va="bottom", fontsize=8, color=SECONDARY,
    )  # fmt: skip
    for ev in events.itertuples():
        if ev.offset <= tmax:
            ax2.text(
                (ev.onset + ev.offset) / 2, top * 0.99, ev.label,
                ha="center", va="top", fontsize=8, color=SECONDARY,
            )  # fmt: skip
    ax2.set_xlim(0, tmax)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Envelope (mV)")
    _style_axes(ax2)
    handles = [
        Line2D([0], [0], color=SERIES[0], lw=1.5),
        Patch(facecolor=SERIES[1], alpha=0.16),
    ]
    fig.legend(
        handles, ["Linear envelope (5 Hz low-pass)", "Detected activation"],
        loc="outside lower center", ncol=2, frameon=False, fontsize=8,
    )  # fmt: skip
    return fig


def plot_scenarios(
    results: pd.DataFrame,
    *,
    chance: float | None = None,
    title: str = "Accuracy depends on how you validate",
) -> Figure:
    """Grouped bars: models on x, one bar per ``(scheme, normalization)`` scenario."""
    combos = list(dict.fromkeys(zip(results["scheme"], results["normalization"], strict=True)))
    if len(combos) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} scenarios can be drawn")
    models = list(dict.fromkeys(results["model"]))
    fig = _figure(8.6, 4.4)
    ax = fig.subplots()
    n, group_w = len(combos), 0.66
    bar_w = group_w / n
    values = np.zeros((n, len(models)))
    for i, (scheme, norm) in enumerate(combos):
        sub = results[(results["scheme"] == scheme) & (results["normalization"] == norm)]
        acc = sub.set_index("model").loc[models, "accuracy"].to_numpy() * 100
        values[i] = acc
        x = np.arange(len(models)) + (i - (n - 1) / 2) * bar_w
        ax.bar(
            x, acc, width=bar_w * 0.88, color=SERIES[i],
            label=SCENARIO_LABELS.get((scheme, norm), f"{scheme} / {norm}"),
        )  # fmt: skip
    best = int(values[-1].argmax())  # label only the model that the story is about
    for i in range(n):
        x_label = best + (i - (n - 1) / 2) * bar_w
        ax.text(
            x_label, values[i, best] + 1.2, f"{values[i, best]:.0f}%",
            ha="center", va="bottom", fontsize=8, color=SECONDARY,
        )  # fmt: skip
    right = len(models) - 0.5
    if chance is not None:
        right += 0.4  # blank strip at the right so the label never sits on top of a bar
        ax.axhline(chance * 100, color=SECONDARY, lw=0.75)
        ax.text(
            right, chance * 100 + 1.2, f"chance ({chance * 100:.0f}%)",
            ha="right", va="bottom", fontsize=8, color=SECONDARY,
        )  # fmt: skip
    ax.set_xlim(-0.5, right)
    ax.set_xticks(range(len(models)), [MODEL_LABELS.get(m, m) for m in models])
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.set_ylabel("Accuracy")
    _style_axes(ax)
    ax.tick_params(axis="x", length=0)
    _title(ax, title)
    fig.legend(loc="outside lower center", ncol=n, frameon=False, fontsize=8)
    return fig


def plot_confusion(cm: pd.DataFrame, *, title: str = "Confusion matrix") -> Figure:
    """Pooled confusion matrix; colour = share of the true class, text = counts."""
    counts = cm.to_numpy(dtype=float)
    share = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)
    fig = _figure(5.6, 4.6)
    ax = fig.subplots()
    image = ax.imshow(share, cmap=BLUE_CMAP, vmin=0, vmax=1)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(
                j, i, f"{int(counts[i, j])}", ha="center", va="center", fontsize=9,
                color="#ffffff" if share[i, j] > 0.55 else PRIMARY,
            )  # fmt: skip
    ax.set_xticks(range(len(cm.columns)), list(cm.columns))
    ax.set_yticks(range(len(cm.index)), list(cm.index))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    _style_axes(ax, grid_axis=None)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    _title(ax, title)
    cbar = fig.colorbar(image, ax=ax, shrink=0.8)
    cbar.set_label("Share of true class", color=SECONDARY, fontsize=8)
    cbar.ax.tick_params(colors=MUTED, labelcolor=SECONDARY, labelsize=8)
    cbar.outline.set_visible(False)
    return fig


def plot_eeg_erd(
    freqs: np.ndarray,
    psd_rest: np.ndarray,
    psd_task: np.ndarray,
    erd_by_channel: Mapping[str, float],
    *,
    channel: str,
    band: tuple[float, float] = (8.0, 13.0),
    fmax: float = 40.0,
) -> Figure:
    """Rest-vs-task PSD of one channel (left) and alpha ERD % for every channel (right)."""
    fig = _figure(10, 3.8)
    ax1, ax2 = fig.subplots(1, 2, width_ratios=[1.5, 1])
    keep = (freqs >= 1) & (freqs <= fmax)
    ax1.semilogy(freqs[keep], psd_rest[keep], color=SERIES[0], lw=1.5, label="Rest")
    ax1.semilogy(freqs[keep], psd_task[keep], color=SERIES[1], lw=1.5, label="Task")
    ax1.axvspan(*band, color=GRID, alpha=0.9, lw=0, zorder=0)
    ax1.text(
        sum(band) / 2, 0.02, "alpha", ha="center", va="bottom", fontsize=8, color=SECONDARY,
        transform=ax1.get_xaxis_transform(),
    )  # fmt: skip
    ax1.set_xlim(1, fmax)
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("PSD (µV²/Hz)")
    _style_axes(ax1)
    _title(ax1, f"Power spectrum, channel {channel}")
    ax1.legend(loc="upper right", frameon=False, fontsize=8)

    names = list(erd_by_channel)
    vals = np.array([erd_by_channel[k] for k in names], dtype=float)
    ax2.bar(range(len(names)), vals, width=0.4, color=SERIES[1])  # orange = task, as on the left
    ax2.axhline(0, color=BASELINE, lw=0.75)
    for i, v in enumerate(vals):
        ax2.text(
            i, v + (-2 if v < 0 else 2), _minus(f"{v:.0f}%"), ha="center",
            va="top" if v < 0 else "bottom", fontsize=8, color=SECONDARY,
        )  # fmt: skip
    ax2.set_xticks(range(len(names)), names)
    low = min(0.0, float(np.nanmin(vals)))
    ax2.set_ylim(low * 1.25 - 5, max(5.0, float(np.nanmax(vals)) * 1.25 + 5))
    ax2.set_ylabel("Alpha ERD (%)")
    _style_axes(ax2)
    ax2.tick_params(axis="x", length=0)
    _title(ax2, "Task vs rest (negative = desynchronisation)")
    return fig
