import numpy as np
import pytest

from biosigkit.force import longest_run_in_range, steadiness_cv, time_in_range, tracking_error


def test_time_and_longest_run_in_range():
    force = np.array([0, 5, 5, 5, 0, 5, 5, 0, 0, 5], dtype=float)  # 10 Hz -> 0.1 s per sample
    assert time_in_range(force, 10, 4, 6) == pytest.approx(0.6)
    assert longest_run_in_range(force, 10, 4, 6) == pytest.approx(0.3)
    assert longest_run_in_range(force, 10, 100, 200) == 0.0


def test_tracking_error_known_values():
    target = np.full(100, 10.0)
    err = tracking_error(target + 1.0, target)
    assert err == {
        "rmse": pytest.approx(1.0),
        "mae": pytest.approx(1.0),
        "bias": pytest.approx(1.0),
    }
    alternating = tracking_error(target + np.tile([-2.0, 2.0], 50), target)
    assert alternating["rmse"] == pytest.approx(2.0)
    assert alternating["bias"] == pytest.approx(0.0)


def test_steadiness_cv():
    assert steadiness_cv(np.full(10, 3.0)) == 0.0
    assert np.isnan(steadiness_cv(np.array([-1.0, 1.0])))  # zero mean
    assert steadiness_cv(np.array([9.0, 11.0])) == pytest.approx(np.std([9, 11], ddof=1) / 10)
