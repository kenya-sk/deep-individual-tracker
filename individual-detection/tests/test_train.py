import numpy as np
import pytest
from detector.constants import FRAME_CHANNEL, LOCAL_IMAGE_SIZE
from detector.train import hard_negative_mining, under_sampling


def test_hard_negative_mining():
    X = np.ones((5, LOCAL_IMAGE_SIZE, LOCAL_IMAGE_SIZE, FRAME_CHANNEL))
    y = np.array([0.32, 0.34, 0.11, 0.92, 0.99], dtype="float32")
    loss_array = np.array([0.1, 0.2, 0.3, 5, 13])  # mean loss = 3.72
    X_hard, y_hard = hard_negative_mining(X, y, loss_array, weight=3)
    assert len(X_hard) == 1
    assert len(y_hard) == 1
    assert y_hard[0] == pytest.approx(0.99)


def test_under_sampling():
    local_iamge_array = np.ones((5, LOCAL_IMAGE_SIZE, LOCAL_IMAGE_SIZE, FRAME_CHANNEL))
    density_array = np.array([0.32, 0.34, 0.11, 0.92, 0.99], dtype="float32")
    threshold = 0.5
    X_under, y_under = under_sampling(local_iamge_array, density_array, threshold)
    assert len(X_under) == 4
    assert len(y_under) == 4
    assert np.max(y_under) == pytest.approx(0.99)
    assert np.min(y_under) < threshold
