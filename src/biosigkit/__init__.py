"""biosigkit — EMG/EEG/force signal processing, feature tables, leakage-safe ML and SQL storage."""

from importlib.metadata import PackageNotFoundError, version

from .eeg import eeg_features
from .emg import emg_features
from .io import read_csv, read_events, write_csv
from .ml import compare_scenarios, cross_validate, within_group_zscore
from .preprocess import EEG_PRESET, EMG_PRESET, PreprocessConfig, preprocess
from .signals import Signal
from .storage import connect
from .synthetic import make_cohort, make_eeg, make_session
from .tables import trial_feature_table, window_feature_table
from .windowing import make_epochs, sliding_windows

try:
    __version__ = version("biosigkit")
except PackageNotFoundError:  # running from a source checkout that was never installed
    __version__ = "0.0.0+local"

__all__ = [
    "EEG_PRESET",
    "EMG_PRESET",
    "PreprocessConfig",
    "Signal",
    "compare_scenarios",
    "connect",
    "cross_validate",
    "eeg_features",
    "emg_features",
    "make_cohort",
    "make_eeg",
    "make_epochs",
    "make_session",
    "preprocess",
    "read_csv",
    "read_events",
    "sliding_windows",
    "trial_feature_table",
    "window_feature_table",
    "within_group_zscore",
    "write_csv",
]
