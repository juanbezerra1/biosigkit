import dataclasses

import numpy as np
import pandas as pd
import pytest

from biosigkit import Signal


def test_1d_data_becomes_single_channel_with_default_name():
    sig = Signal(np.arange(10), fs=100)
    assert sig.data.shape == (1, 10)
    assert sig.channels == ("ch1",)
    assert sig.duration == pytest.approx(0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"data": np.zeros((2, 5)), "fs": 0},
        {"data": np.zeros((2, 5)), "fs": float("nan")},
        {"data": np.zeros((2, 5)), "fs": 10, "channels": ("a",)},
        {"data": np.zeros((2, 5)), "fs": 10, "channels": ("a", "a")},
        {"data": np.zeros((2, 3, 5)), "fs": 10},
    ],
)
def test_invalid_construction_raises(kwargs):
    with pytest.raises(ValueError):
        Signal(**kwargs)


def test_signal_is_immutable_and_with_data_returns_a_copy():
    sig = Signal(np.ones((1, 4)), fs=10, meta={"a": 1})
    with pytest.raises(dataclasses.FrozenInstanceError):
        sig.fs = 20  # type: ignore[misc]
    new = sig.with_data(np.zeros((1, 4)))
    assert sig.data.sum() == 4 and new.data.sum() == 0
    new.meta["a"] = 2
    assert sig.meta["a"] == 1


def test_pick_keeps_requested_order_and_rejects_unknown():
    sig = Signal(np.arange(6).reshape(3, 2), fs=10, channels=("a", "b", "c"))
    picked = sig.pick(["c", "a"])
    assert picked.channels == ("c", "a")
    np.testing.assert_array_equal(picked.data, [[4, 5], [0, 1]])
    with pytest.raises(KeyError):
        sig.pick(["zzz"])


def test_crop_uses_absolute_time_and_updates_t0():
    sig = Signal(np.arange(100), fs=10, t0=5.0)  # covers 5.0 .. 14.9 s
    out = sig.crop(7.0, 9.0)
    assert out.n_samples == 20
    assert out.t0 == pytest.approx(7.0)
    assert out.data[0, 0] == 20
    with pytest.raises(ValueError):
        sig.crop(9.0, 7.0)


def test_frame_roundtrip_preserves_data_fs_and_t0():
    sig = Signal(np.random.default_rng(0).normal(size=(2, 50)), fs=250, channels=("x", "y"), t0=1.5)
    back = Signal.from_frame(sig.to_frame(), time_col="time")
    np.testing.assert_allclose(back.data, sig.data)
    assert back.fs == pytest.approx(250)
    assert back.t0 == pytest.approx(1.5)
    assert back.channels == ("x", "y")


def test_from_frame_validates_time_column_and_fs():
    df = pd.DataFrame({"time": np.arange(100) / 100, "a": np.zeros(100)})
    with pytest.raises(ValueError, match="does not match"):
        Signal.from_frame(df, fs=500, time_col="time")
    bad = df.copy()
    bad.loc[10, "time"] = 0.0
    with pytest.raises(ValueError, match="strictly increasing"):
        Signal.from_frame(bad, time_col="time")
    with pytest.raises(ValueError, match="provide fs"):
        Signal.from_frame(df[["a"]])
