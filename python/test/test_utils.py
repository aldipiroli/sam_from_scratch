import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from utils.misc import downsample_mask, get_prompt_from_gtmask


def test_get_prompt_from_gtmask():
    b = 2
    mask = torch.zeros(b, 244, 244)
    mask[0, 10, 10] = 1
    mask[0, 10, 11] = 2
    selected_prompts, selected_masks, selected_classes = get_prompt_from_gtmask(mask)
    assert selected_prompts.shape == (b, 2)
    assert selected_masks.shape == mask.shape
    assert selected_classes.shape == (b,)
    for c in selected_classes:
        assert c in [0, 1, 2]


def test_mask_downsampling():
    b = 2
    target_h, target_w = 56, 56
    mask = torch.zeros(b, 244, 244)
    mask[:, :50, :50] = 1
    mask_down = downsample_mask(mask, (target_h, target_w))
    assert mask_down.shape == (b, target_h, target_w)


if __name__ == "__main__":
    test_get_prompt_from_gtmask()
    test_mask_downsampling()
    print("Tests passed!")
