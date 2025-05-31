import torch
import torch.nn as nn
from utils.misc import downsample_mask


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, pred, gt):
        b = pred.shape[0]
        pred = pred.reshape(b, -1)
        gt = gt.reshape(b, -1)
        assert pred.shape == gt.shape

        intersection = (pred * gt).sum(dim=1)
        union = pred.sum(dim=1) + gt.sum(dim=1)

        dice = (2 * intersection + self.eps) / (union + self.eps)
        loss = 1 - dice
        return loss.mean()


class SAMLoss(nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, gt_masks, pred_masks, iou):
        b, num_masks, pred_h, pred_w = pred_masks.shape
        gt_mask_down = downsample_mask(gt_masks, target_dim=(pred_h, pred_w)).to(pred_masks.device)
        dice_loss_fn = DiceLoss()
        mask_losses = []
        for i in range(num_masks):
            curr_loss = dice_loss_fn(pred_masks[:, i, :, :], gt_mask_down)
            mask_losses.append(curr_loss)
        mask_losses = torch.stack(mask_losses, 0)
        loss_idx = torch.argmin(mask_losses)
        loss = mask_losses[loss_idx]
        return loss
