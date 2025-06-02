import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model.loss_functions import DiceLoss, compute_iou_between_masks


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


def tes_compute_iou_between_masks():
    b = 1
    # iou = 1
    pred = torch.zeros(b, 3, 56, 56)
    gt = torch.zeros(b, 1, 56, 56)
    pred[:, :, :10, :10] = 1
    gt[:, :, :10, :10] = 1
    iou = compute_iou_between_masks(gt, pred)
    assert torch.equal(iou, torch.tensor([[1, 1, 1]]))

    # iou = 0
    pred = torch.zeros(b, 3, 56, 56)
    gt = torch.zeros(b, 1, 56, 56)
    pred[:, :, :10, :10] = 1
    gt[:, :, 10:, 10:] = 1
    iou = compute_iou_between_masks(gt, pred)
    assert torch.equal(iou, torch.tensor([[0, 0, 0]]))

    # iou = 0.5
    pred = torch.zeros(b, 3, 56, 56)
    gt = torch.zeros(b, 1, 56, 56)
    pred[:, :, :5, :] = 1
    gt[:, :, :10, :] = 1
    iou = compute_iou_between_masks(gt, pred)
    assert torch.equal(iou, torch.tensor([[0.5, 0.5, 0.5]]))


if __name__ == "__main__":
    test_dice_loss()
    tes_compute_iou_between_masks()
    print("Tests passed!")
