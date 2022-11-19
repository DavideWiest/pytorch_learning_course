import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm
import os, sys
from pathlib import Path

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict, List

from PIL import Image

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
EPOCHS = 3
CONV_BLOCK2_OUTPUT_SHAPE = 7

device = "cuda" if torch.cuda.is_available() else "cpu"

img_train = Path("C:/Users/DavWi/OneDrive/Desktop/storage/ml_datasets/foods/train")
img_test = Path("C:/Users/DavWi/OneDrive/Desktop/storage/ml_datasets/foods/test")

img_train_list = list(img_train.glob("*/*.jpg"))
img_test_list = list(img_test.glob("*/*.jpg"))

data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=img_train, transform=data_transform, target_transform=None)
test_data = datasets.ImageFolder(root=img_test, transform=data_transform, target_transform=None)

# print(train_data.class_to_idx)
# print(train_data.classes)

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), shuffle=True)

img, label = next(iter(train_data))

def find_classes(directory: str):
    class_names_found = sorted([entry.name for entry in list(os.scandir(directory)) if entry.is_dir()])

    if not class_names_found:
        raise FileNotFoundError("Couldnt find any classes")

    class_idx = {i: c for i, c in enumerate(class_names_found)}
    # print(class_idx)
    return class_names_found, class_idx

find_classes(img_train)


class ImageFolderCustom(Dataset):
    def __init__(self, target_dir: str, transform=None):
        self.paths = list(Path(target_dir).glob("*/*.jpg"))

        self.transform = transform
        self.classes, self.class_to_idx = find_classes(target_dir)

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx


train_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    # transforms.RandomHorizontalFlip(p=0.5), # usually only for train_transform
    transforms.ToTensor()
])

custom_train_data = ImageFolderCustom(img_train, train_transform)
custom_test_data = ImageFolderCustom(img_test, test_transform)

# print(custom_train_data.classes)
# print(custom_test_data.class_to_idx)

train_dataloader = DataLoader(
    custom_train_data,
    BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

test_dataloader = DataLoader(
    custom_test_data,
    BATCH_SIZE,
    shuffle=False, # usually only for train_dataloader
    num_workers=NUM_WORKERS
)

img, label = next(iter(train_dataloader))

print(img.shape, "\n", label.shape)


