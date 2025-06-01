import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model.loss_functions import DiceLoss


def test_dice_loss():
    loss_fn = DiceLoss()
    b = 1

    # iou = 1
    pred = torch.zeros(b, 56, 56)
    gt = torch.zeros(b, 56, 56)
    pred[:, :10, :10] = 1
    gt[:, :10, :10] = 1
    loss = loss_fn(pred, gt)
    assert loss == 0

    # iou = 0
    pred = torch.zeros(b, 56, 56)
    gt = torch.zeros(b, 56, 56)
    pred[:, :10, :10] = 1
    gt[:, 10:, 10:] = 1
    loss = loss_fn(pred, gt)
    assert loss == 1

    pred = torch.zeros(b, 56, 56)
    gt = torch.zeros(b, 56, 56)
    pred[:, :10, :10] = 1
    gt[:, :5, :5] = 1
    loss = loss_fn(pred, gt)
    assert loss == 0.6


if __name__ == "__main__":
    test_dice_loss()
    print("Tests passed!")
