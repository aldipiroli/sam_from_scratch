import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pascal_voc_dataset.pascal_voc_dataset import PascalVOCDataset
from torch.utils.data import DataLoader


def test_pascal_voc_dataset(skip=True):
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


if __name__ == "__main__":
    test_pascal_voc_dataset(skip=False)
    print("Tests passed!")
