import os
from pathlib import Path

from pascal_voc_dataset.download_pascal_voc_dataset import download_pascal_voc_2012_dataset
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SimpleTransform:
    def __init__(self, size=(256, 256)):
        self.image_transform = T.Compose(
            [
                T.Resize(size),
                T.ToTensor(),
            ]
        )
        self.mask_transform = T.Compose(
            [
                T.Resize(size, interpolation=Image.NEAREST),
                T.PILToTensor(),
            ]
        )

    def __call__(self, image, mask):
        return self.image_transform(image), self.mask_transform(mask).long().squeeze(0)


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
        self.transforms = SimpleTransform()

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

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask
