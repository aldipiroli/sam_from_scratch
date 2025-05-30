import logging
import os
from datetime import datetime

import numpy as np
import torch
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from pascal_voc_dataset.pascal_voc_dataset import decode_voc_mask

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



def save_image_with_mask(image, mask, filename, alpha=0.5):
    assert image.ndim == 3 and image.shape[0] == 3, "Image must be of shape (3, n, h)"
    assert mask.ndim == 2 and mask.shape == image.shape[1:], "Mask must be of shape (n, h)"

    image_np = image.detach().cpu().float().numpy()
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    image_np = np.transpose(image_np, (1, 2, 0))  # (H, W, 3)

    # Convert mask to RGB
    mask_rgb = decode_voc_mask(mask) / 255.0  # (H, W, 3), float in [0, 1]

    # Create overlay
    overlay = image_np * (1 - alpha) + mask_rgb * alpha
    overlay = np.clip(overlay, 0, 1)

    # Concatenate horizontally
    panel = np.concatenate([image_np, mask_rgb, overlay], axis=1)

    # Save to file
    plt.imsave(filename, panel)

