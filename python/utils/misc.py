import logging
import os
from datetime import datetime

import matplotlib.gridspec as gridspec
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


def get_overlayer_image(image_np, mask, alpha=0.7):
    if image_np.max() > 1.0:
        image_np /= 255.0
    mask_rgb = decode_voc_mask(mask) / 255.0  # (H, W, 3), in [0, 1]
    overlay = np.clip(image_np * (1 - alpha) + mask_rgb * alpha, 0, 1)
    return overlay, mask_rgb


def plot_mask_predictions(
    image,
    original_masks,
    target_gt_mask,
    pred_masks,
    pred_ious,
    actual_iou,
    prompt=None,
    filename="tmp.png",
    img_size=(224, 224),
):
    image = np.transpose(image.detach().cpu().float().numpy(), (1, 2, 0))
    original_masks = original_masks.detach().cpu().float().numpy()
    target_gt_mask = target_gt_mask.detach().cpu().float().numpy()
    pred_masks = pred_masks.detach().cpu().float().numpy()
    pred_ious = pred_ious.detach().cpu().float().numpy()
    actual_iou = actual_iou.detach().cpu().float().numpy()
    n_preds = pred_masks.shape[0]

    if prompt is not None:
        prompt_np = prompt.detach().cpu()
        px, py = int(prompt_np[1] * img_size[0]), int(prompt_np[0] * img_size[1])
        prompt_output = (
            f"Prompt value: {original_masks[px, py]:.2f}, "
            f"Unique: {np.unique(original_masks)}, Sum {np.sum(original_masks)}"
        )
    else:
        px, py = None, None
        prompt_output = f"Unique: {np.unique(original_masks)}, Sum {np.sum(original_masks)}"

    n_uniques = len(np.unique(original_masks))
    max_cols = max(3 + n_preds, n_uniques)  # Enough columns for all pred masks and unique masks
    fig = plt.figure(figsize=(5 * max_cols, 10))
    gs = gridspec.GridSpec(2, max_cols, figure=fig)

    # Input Image
    overlay, _ = get_overlayer_image(image, original_masks, alpha=0.7)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(overlay)
    if prompt is not None:
        ax0.plot(px, py, "x", markersize=10, color="red")
    ax0.set_title("Input Image")
    ax0.axis("off")

    # Original Mask
    ax1 = fig.add_subplot(gs[0, 1])
    m = ax1.imshow(original_masks)
    if prompt is not None:
        ax1.plot(px, py, "x", markersize=10, color="red")
    ax1.set_title("Original Mask")
    ax1.axis("off")
    fig.colorbar(m, ax=ax1)

    # GT Mask
    ax2 = fig.add_subplot(gs[0, 2])
    im_gt = ax2.imshow(target_gt_mask, cmap="viridis")
    if prompt is not None:
        ax2.plot(px, py, "x", markersize=10, color="red")
    ax2.set_title("Target GT Mask")
    ax2.axis("off")
    fig.colorbar(im_gt, ax=ax2)

    # Predicted Masks
    for i in range(n_preds):
        ax = fig.add_subplot(gs[0, 3 + i])
        im_pred = ax.imshow(pred_masks[i], cmap="viridis")
        ax.set_title(f"M {i+1}, Pred IoU {pred_ious[i].item():.2f} / {actual_iou[i].item():.2f}")
        ax.axis("off")
        fig.colorbar(im_pred, ax=ax)

    # Unique value masks in original
    for i, u in enumerate(np.unique(original_masks)):
        ax = fig.add_subplot(gs[1, i])
        curr_mask = original_masks == u
        ax.set_title(f"Value {u}")
        ax.imshow(curr_mask, cmap="viridis")
        if prompt is not None:
            ax.plot(px, py, "x", markersize=10, color="red")
        fig.colorbar(ax.images[0], ax=ax)

    fig.suptitle(prompt_output, fontsize=16)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    print(f"Saved Image: {filename}")
    plt.close()


def get_prompt_from_gtmask(mask, deterministic=False):
    b, h, w = mask.shape
    device = mask.device

    selected_prompts = []
    selected_classes = []
    selected_masks = []

    for i in range(b):
        unique_classes = torch.unique(mask[i])
        if deterministic:
            selected_class = unique_classes[0]
        else:
            selected_class = unique_classes[torch.randint(len(unique_classes), (1,)).item()]
        selected_classes.append(selected_class)

        class_mask = mask[i] == selected_class
        indices = torch.nonzero(class_mask, as_tuple=False)

        if deterministic:
            selected_index = indices[0]
        else:
            selected_index = indices[torch.randint(len(indices), (1,)).item()]
        selected_prompts.append(selected_index)
        selected_masks.append(class_mask)

    selected_prompts = torch.stack(selected_prompts, dim=0).unsqueeze(1)  # Note: here n_prompts=1
    selected_masks = torch.stack(selected_masks, dim=0)
    selected_classes = torch.tensor(selected_classes, device=device)
    return (
        selected_prompts,  # (b,n_prompts,2)
        selected_masks,  # (b,h,w)
        selected_classes,  # (b,)
    )


def downsample_mask(mask, target_dim):
    mask_down = F.interpolate(mask.unsqueeze(1).float(), size=(target_dim[0], target_dim[1]), mode="nearest").squeeze(1)
    return mask_down


def get_simple_data(b=1, h=224, w=224):
    img = torch.zeros(b, 3, h, w)
    mask = torch.zeros(b, h, w)
    step = 75

    img[:, :, :step, :] = 0
    mask[:, :step, :] = 0

    img[:, :, step : 2 * step, :] = 1
    mask[:, step : 2 * step, :] = 1

    img[:, :, 2 * step :, :] = 2
    mask[:, 2 * step :, :] = 2
    plot_tensor_to_file(mask[0], filename="original_mask.png")
    return img, mask


def plot_tensor_to_file(tensor, p=[-1, -1], v=-1, filename="tmp"):
    tensor = tensor.cpu()
    if tensor.ndim == 3:
        tensor = tensor.permute(1, 2, 0)

    print("Uniques", torch.unique(tensor))
    tensor = tensor.numpy()

    plt.figure()
    plt.imshow(tensor)

    if p is not None:
        plt.plot(p[0], p[1], "x", markersize=10, color="red")

    # Add axis labels
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(p)

    plt.savefig(f"tmp/{filename}.png")
    plt.close()
