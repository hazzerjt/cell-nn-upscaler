import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
import numpy as np

class cellDataset(Dataset):
    def __init__(self, lowres_csv, highres_csv, highres_dir, lowres_dir, transform=None):
        self.lowres_csv = pd.read_csv(lowres_csv)
        self.highres_csv = pd.read_csv(highres_csv)
        self.highres_dir = highres_dir
        self.lowres_dir = lowres_dir

    def __len__(self):
        return len(self.lowres_csv)
    def __getitem__(self, index):
        highres_path = os.path.join(self.highres_dir, self.highres_csv.iloc[index, 0])
        lowres_path = os.path.join(self.lowres_dir, self.lowres_csv.iloc[index, 0])

        highres_image = Image.open(highres_path)
        lowres_image = Image.open(lowres_path)

        highres_image = np.array(highres_image)
        lowres_image = np.array(lowres_image)

        highres_image = highres_image.astype(float)
        lowres_image = lowres_image.astype(float)

        highres_image = torch.Tensor(highres_image)
        lowres_image = torch.Tensor(lowres_image)

        highres_image = highres_image[None, :, :]
        lowres_image = lowres_image[None, :, :]



        return {
            'highres': highres_image,
            'lowres': lowres_image
        }
