import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pascal_voc_dataset.pascal_voc_dataset import decode_voc_mask
from PIL import Image


def get_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    return device


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"log_{now}.log")
    logger = logging.getLogger(f"logger_{now}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def plot_single_image(tensor, filename):
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)

    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.save(filename)


def plot_image_with_mask(image, mask, prompt=None, filename="tmp.png", alpha=0.5, overlay=False):
    assert image.ndim == 3 and image.shape[0] == 3, "Image must be (3, H, W)"
    assert mask.ndim == 2 and mask.shape == image.shape[1:], "Mask must match image spatial size"

    # Normalize and format image
    image_np = image.detach().cpu().float().numpy()
    if image_np.max() > 1.0:
        image_np /= 255.0
    image_np = np.transpose(image_np, (1, 2, 0))  # (H, W, 3)

    # Prepare mask and overlay
    mask_rgb = decode_voc_mask(mask) / 255.0  # (H, W, 3), in [0, 1]
    overlay = np.clip(image_np * (1 - alpha) + mask_rgb * alpha, 0, 1)

    if prompt is not None:
        x, y = prompt[0].item(), prompt[1].item()
        fig, axs = plt.subplots(1, 3, figsize=(10, 10))
        for ax, img in zip(axs, [image_np, mask_rgb, overlay]):
            ax.imshow(img)
            ax.plot(x, y, "x", markersize=10, color="red")
            ax.axis("off")
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        panel = np.concatenate([image_np, mask_rgb, overlay], axis=1)
        plt.imsave(filename, panel)


def plot_mask_predictions(image, gt_mask, masks, filename="tmp.png", prompt=None, alpha=0.5):
    image_np = image.detach().cpu().float().numpy()
    gt_mask = gt_mask.detach().cpu().float().numpy()
    masks = masks.detach().cpu().float().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))

    masks = masks[:3]
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    axs[0].imshow(image_np)
    if prompt is not None:
        x, y = prompt[0].item(), prompt[1].item()
        axs[0].plot(x, y, "x", markersize=10, color="red")
    axs[0].axis("off")

    axs[1].imshow(gt_mask)
    if prompt is not None:
        x, y = prompt[0].item(), prompt[1].item()
        axs[1].plot(x, y, "x", markersize=10, color="red")
    axs[1].axis("off")

    for i, (ax, mask) in enumerate(zip(axs[2:], masks)):
        ax.imshow(mask)
        ax.axis("off")

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(current_time, fontsize=16)

    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    print(f"Saved Image: {filename}")
    plt.close()


def get_prompt_from_gtmask(mask, deterministic=False):
    b = mask.shape[0]
    selected_prompts = []
    selected_classes = []
    selected_masks = []

    for curr_batch in range(b):
        unique = torch.unique(mask[curr_batch].flatten())

        if deterministic:
            selected_class = unique[0]
        else:
            random_index = torch.randint(0, len(unique), (1,)).item()
            selected_class = unique[random_index]
        selected_classes.append(selected_class)

        possible_prompts = mask[curr_batch] == selected_class
        possible_prompts_indices = torch.nonzero(possible_prompts)

        if deterministic:
            prompt_idx = 0
        else:
            prompt_idx = torch.randperm(possible_prompts_indices.size(0))[0]
        selected_prompts.append(possible_prompts_indices[prompt_idx])

        curr_mask = mask[curr_batch] == selected_class
        selected_masks.append(curr_mask)

    selected_prompts = torch.stack(selected_prompts, 0)
    selected_classes = torch.tensor(selected_classes).int()
    selected_masks = torch.stack(selected_masks, 0)
    return selected_prompts, selected_masks, selected_classes


def downsample_mask(mask, target_dim):
    mask_down = F.interpolate(mask.unsqueeze(1).float(), size=(target_dim[0], target_dim[1]), mode="nearest").squeeze(1)
    return mask_down
