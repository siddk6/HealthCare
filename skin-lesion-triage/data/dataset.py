"""PyTorch Dataset for HAM10000, built from the split CSVs prepare_splits.py writes."""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CLASS_TO_IDX, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class HAM10000Dataset(Dataset):
    """Reads from a split CSV with at minimum: image_path, dx columns."""

    def __init__(self, csv_path: str, train: bool = False, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform or build_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = CLASS_TO_IDX[row["dx"]]
        return image, label

    def class_counts(self):
        return self.df["dx"].value_counts().reindex(CLASS_TO_IDX.keys(), fill_value=0)
