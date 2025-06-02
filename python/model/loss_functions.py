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
    def __init__(self, loss_weights):
        super(SAMLoss, self).__init__()
        self.loss_weights = loss_weights

    def forward(self, gt_masks, pred_masks, pred_iou):
        b, num_masks, pred_h, pred_w = pred_masks.shape
        gt_mask_down = downsample_mask(gt_masks, target_dim=(pred_h, pred_w)).to(pred_masks.device)
        gt_masks = gt_mask_down.float().unsqueeze(1)  # (B, 1, H, W)

        bce_losses = F.binary_cross_entropy_with_logits(
            pred_masks, gt_masks.expand(-1, num_masks, -1, -1), reduction="none"
        )  # (n, num_masks, h, w)
        bce_losses = bce_losses.mean(dim=(2, 3))  # (B, N)
        min_bce_loss, _ = bce_losses.min(dim=1)  # (B,)

        # iou loss
        mse_loss_fn = nn.MSELoss()
        iou_gt_pred = compute_iou_between_masks(gt_masks, pred_masks)
        iou_loss = mse_loss_fn(iou_gt_pred, pred_iou)

        tot_loss = (min_bce_loss.sum() * self.loss_weights["mask_pred_loss_weight"]) + (
            iou_loss.sum() * self.loss_weights["iou_loss_weight"]
        )
        return tot_loss


def compute_iou_between_masks(gt_mask, pred_masks, threshold=0.5):
    pred_mask_binary = (pred_masks > threshold).bool()
    gt_mask = gt_mask.bool()

    gt_mask_exp = gt_mask.expand(-1, pred_mask_binary.shape[0], -1, -1)
    intersection = (gt_mask_exp & pred_mask_binary).sum(dim=(2, 3)).float()  # (B, N)
    union = (gt_mask_exp | pred_mask_binary).sum(dim=(2, 3)).float()  # (B, N)

    iou = intersection / (union + 1e-6)
    return iou
