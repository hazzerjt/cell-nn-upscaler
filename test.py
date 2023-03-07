import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class cellDataset(Dataset):
    def __init__(self, lowres_csv, highres_csv, highres_dir, lowres_dir, transform=None):
        self.lowres_csv = pd.read_csv(lowres_csv)
        self.highres_csv = pd.read_csv(highres_csv)
        self.highres_dir = highres_dir
        self.lowres_dir = lowres_dir
        self.transform = transform

    def __len__(self):
        return len(self.lowres_csv)
    def __getitem__(self, index):
        highres_path = os.path.join(self.highres_dir, self.highres_csv.iloc[index, 0])
        lowres_path = os.path.join(self.lowres_dir, self.lowres_csv.iloc[index, 0])

        highres_image = read_image(highres_path)
        lowres_image = read_image(lowres_path)

        if self.transform:
            highres_image_transformed = self.transform(highres_image)
            lowres_image_transformed = self.transform(lowres_image)

        return {
            'highres': highres_image_transformed,
            'lowres': lowres_image_transformed
        }
