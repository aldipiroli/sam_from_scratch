import torch
import torch.nn as nn


class PixelReconstructionLoss(nn.Module):
    def __init__(self):
        super(PixelReconstructionLoss, self).__init__()

    def forward(self, preds, gt, mask, normalize=False, use_masking=False):
        if not use_masking:
            mask = torch.ones_like(mask)
        if normalize:
            mean = torch.mean(gt, dim=-1, keepdim=True)
            std = torch.std(gt, dim=-1, keepdim=True) + 1e-6
            target = (gt - mean) / std
        else:
            target = gt

        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss
