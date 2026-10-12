from __future__ import annotations

import json

import numpy as np

from platoon_control.data import MatDelayDataset
from platoon_control.ngsim import NGSIMStore, NGSIMTrajectoryDataset, NGSIMWaveDataset


def test_ngsim_manifest_has_guarded_temporal_splits_and_class_mix() -> None:
    store = NGSIMStore("artifacts/ngsim")
    train = store.split("train")
    validation = store.split("validation")
    test = store.split("test")
    assert validation.start_ms - train.end_ms == 5_000
    assert test.start_ms - validation.end_ms == 5_000
    probabilities = store.manifest["vehicle_class_probabilities"]
    assert np.isclose(sum(probabilities.values()), 1.0)
    assert 0.94 < probabilities["2"] < 0.96


def test_supervised_datasets_have_disjoint_splits_and_expected_labels() -> None:
    delay_train = MatDelayDataset(split="train")
    delay_validation = MatDelayDataset(split="validation")
    delay_test = MatDelayDataset(split="test")
    assert delay_train.indices[-1] < delay_validation.indices[0]
    assert delay_validation.indices[-1] < delay_test.indices[0]

    trajectory = NGSIMTrajectoryDataset(split="validation")[0]
    assert trajectory["history"].shape == (12, 20, 8)
    assert trajectory["future"].shape == (12, 30, 4)
    assert trajectory["front_history"].shape == (8, 20, 6)
    assert set(trajectory["lane_change"].tolist()) <= {0, 1, 2}

    wave = NGSIMWaveDataset(split="test")[0]
    assert wave["condition"].shape == (20, 8)
    assert wave["target_acceleration"].shape == (60,)
    assert float(wave["target_acceleration"].min()) < -1.0
