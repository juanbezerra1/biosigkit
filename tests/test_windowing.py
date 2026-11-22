import numpy as np
import pandas as pd
import pytest

from biosigkit import Signal
from biosigkit.windowing import make_epochs, sliding_windows, validate_events


def _events(rows):
    return pd.DataFrame(rows, columns=["onset", "offset", "label"])


def test_validate_events_sorts_and_keeps_extra_columns():
    ev = _events([(5, 6, "b"), (1, 2, "a")]).assign(level=[2, 1])
    out = validate_events(ev)
    assert list(out["label"]) == ["a", "b"]
    assert list(out["level"]) == [1, 2]


@pytest.mark.parametrize(
    "bad",
    [
        pd.DataFrame({"onset": [1.0], "label": ["a"]}),
        _events([(2, 2, "a")]),
        _events([(1, 3, "a"), (2, 4, "b")]),
    ],
)
def test_validate_events_rejects_bad_tables(bad):
    with pytest.raises(ValueError):
        validate_events(bad)


def test_sliding_windows_shapes_and_times():
    sig = Signal(np.arange(2000).reshape(2, 1000), fs=100, t0=10.0)
    windows, t_start, t_stop = sliding_windows(sig, win_s=1.0, step_s=0.5)
    assert windows.shape == (19, 2, 100)
    np.testing.assert_allclose(t_start[:3], [10.0, 10.5, 11.0])
    np.testing.assert_allclose(t_stop - t_start, 1.0)
    np.testing.assert_array_equal(windows[1, 0], sig.data[0, 50:150])
    assert t_stop[-1] == pytest.approx(sig.t0 + sig.duration)


def test_sliding_windows_validation():
    sig = Signal(np.zeros(100), fs=100)
    with pytest.raises(ValueError, match="longer"):
        sliding_windows(sig, 2.0, 0.5)
    with pytest.raises(ValueError):
        sliding_windows(sig, 0.001, 0.5)


def test_make_epochs_extracts_aligned_segments_and_drops_out_of_range():
    sig = Signal(np.arange(1000, dtype=float), fs=100)  # 10 s, value == sample index
    events = _events([(2.0, 3.0, "a"), (4.0, 5.0, "b"), (9.8, 9.9, "c")])
    ep = make_epochs(sig, events, tmin=-0.5, tmax=1.0)
    assert ep.data.shape == (2, 1, 150)
    assert ep.data[0, 0, 0] == 150  # 2.0 s - 0.5 s -> sample 150
    assert list(ep.events["label"]) == ["a", "b"]
    assert ep.times[0] == pytest.approx(-0.5) and len(ep) == 2
    with pytest.raises(ValueError, match="outside"):
        make_epochs(sig, events, tmin=-0.5, tmax=1.0, on_out_of_range="raise")
    kept = ep.select(np.array([False, True]))
    assert list(kept.events["label"]) == ["b"] and kept.data.shape[0] == 1


def test_make_epochs_rejects_inverted_window():
    with pytest.raises(ValueError):
        make_epochs(Signal(np.zeros(100), 100), _events([(0.2, 0.4, "a")]), tmin=1, tmax=0)
