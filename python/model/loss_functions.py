import torch.nn as nn
import torch.nn.functional as F
from utils.misc import downsample_mask


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.eps = 1e-6

    def intersection(self, x1, x2):
        intersection = (x1 * x2).sum(dim=1)
        return intersection

    def union(self, x1, x2):
        union = x1.sum(dim=1) + x2.sum(dim=1)
        return union

    def forward(self, pred, gt):
        b = pred.shape[0]
        pred = pred.reshape(b, -1)
        gt = gt.reshape(b, -1)
        assert pred.shape == gt.shape

        intersection = self.intersection(pred, gt)
        union = self.union(pred, gt)
        # Note: Dice loss does not substract the intersection (https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient)

        dice = (2 * intersection + self.eps) / (union + self.eps)
        loss = 1 - dice
        return loss.mean()


class SAMLoss(nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, gt_masks, pred_masks, iou):
        b, num_masks, pred_h, pred_w = pred_masks.shape
        gt_mask_down = downsample_mask(gt_masks, target_dim=(pred_h, pred_w)).to(pred_masks.device)

        gt_masks = gt_mask_down.float().unsqueeze(1)  # (B, 1, H, W)

        bce_losses = F.binary_cross_entropy_with_logits(
            pred_masks, gt_masks.expand(-1, num_masks, -1, -1), reduction="none"
        )  # (B, N, H, W)
        bce_losses = bce_losses.mean(dim=(2, 3))  # (B, N)

        min_losses, _ = bce_losses.min(dim=1)  # (B,)
        return min_losses.mean()
