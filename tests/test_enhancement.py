import torch
import pytest
from enhancement.model import DCE_Net
from enhancement.losses import ZeroDCELoss

def test_zero_dce_forward():
    model = DCE_Net(n_iterations=8, base_channels=16)
    x = torch.rand(2, 3, 64, 64)
    out, A = model(x)

    assert out.shape == (2, 3, 64, 64)
    assert A.shape == (2, 24, 64, 64)
    assert out.min() >= 0.0 and out.max() <= 1.0

def test_zero_dce_loss():
    criterion = ZeroDCELoss()
    org = torch.rand(2, 3, 64, 64)
    enhanced = torch.rand(2, 3, 64, 64)
    A = torch.rand(2, 24, 64, 64)

    total_loss, loss_dict = criterion(org, enhanced, A)
    assert total_loss.item() > 0.0
    assert "spatial_loss" in loss_dict
    assert "exposure_loss" in loss_dict
    assert "color_loss" in loss_dict
    assert "tv_loss" in loss_dict
