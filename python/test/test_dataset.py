import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pascal_voc_dataset.pascal_voc_dataset import PascalVOCDataset
from torch.utils.data import DataLoader
from utils.misc import save_image_with_mask


def test_tiny_imagenet_dataset(skip=True):
    if skip:
        return True
    curr_split = "train"
    for curr_split in ["train", "val"]:
        dataset = PascalVOCDataset(root_dir="../data", split=curr_split)
        batch_size = 8
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        it = iter(loader)
        images, masks = next(it)
        assert images.shape == (batch_size, 3, 224, 224)
        assert masks.shape == (batch_size, 224, 224)
        save_image_with_mask(images[0], masks[0], "tmp.png")


if __name__ == "__main__":
    test_tiny_imagenet_dataset(skip=False)
    print("Tests passed!")
