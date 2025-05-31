import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from pascal_voc_dataset.download_pascal_voc_dataset import download_pascal_voc_2012_dataset
from PIL import Image
from torch.utils.data import Dataset

VOC_COLORMAP = np.array(
    [
        [0, 0, 0],  # 0=background
        [128, 0, 0],  # 1=aeroplane
        [0, 128, 0],  # 2=bicycle
        [128, 128, 0],  # 3=bird
        [0, 0, 128],  # 4=boat
        [128, 0, 128],  # 5=bottle
        [0, 128, 128],  # 6=bus
        [128, 128, 128],  # 7=car
        [64, 0, 0],  # 8=cat
        [192, 0, 0],  # 9=chair
        [64, 128, 0],  # 10=cow
        [192, 128, 0],  # 11=diningtable
        [64, 0, 128],  # 12=dog
        [192, 0, 128],  # 13=horse
        [64, 128, 128],  # 14=motorbike
        [192, 128, 128],  # 15=person
        [0, 64, 0],  # 16=potted plant
        [128, 64, 0],  # 17=sheep
        [0, 192, 0],  # 18=sofa
        [128, 192, 0],  # 19=train
        [0, 64, 128],  # 20=tv/monitor
    ],
    dtype=np.uint8,
)


def decode_voc_mask(mask: torch.Tensor) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = mask.astype(np.uint8)

    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_mask = VOC_COLORMAP[mask]
    return rgb_mask


class ViTV16Transforms:
    def __init__(self):
        """
        Matching transforms for: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.ViT_B_16_Weights
        """
        self.image_transform = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(224),
                T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.mask_resize = T.Resize(256, interpolation=T.InterpolationMode.NEAREST)
        self.mask_crop = T.CenterCrop(224)

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = self.image_transform(image)

        mask = np.array(mask, dtype=np.uint8)
        mask = Image.fromarray(mask)
        mask = self.mask_resize(mask)
        mask = self.mask_crop(mask)
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask


class PascalVOCDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        Args:
            root_dir (str): Path to the VOCdevkit directory (e.g., "./data/VOCdevkit/VOC2012")
            split (str): "train" or "val"
            transforms (callable, optional): Transformations to apply to both image and mask
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = ViTV16Transforms()

        if not os.path.isdir(self.root_dir / "VOCdevkit/VOC2012"):
            download_pascal_voc_2012_dataset(self.root_dir)
        self.root_dir = self.root_dir / "VOCdevkit/VOC2012"

        image_set_path = os.path.join(self.root_dir, "ImageSets", "Segmentation", f"{split}.txt")
        with open(image_set_path, "r") as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        self.image_dir = os.path.join(self.root_dir, "JPEGImages")
        self.mask_dir = os.path.join(self.root_dir, "SegmentationClass")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{image_id}.png")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.uint8)
        mask = torch.from_numpy(mask).long()
        mask[mask == 255] = 0  # mask unknown to background (for simplicity)

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask
