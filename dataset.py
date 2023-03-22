import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
import numpy as np
import torchvision.transforms.functional as f

#Implement sharpening and brightness. One at a time
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

        highres_image_mean = highres_image.mean()
        lowres_image_mean = lowres_image.mean()

        highres_image_std = highres_image.std()
        lowres_image_std = lowres_image.std()

        highres_image = highres_image[None, :, :]
        lowres_image = lowres_image[None, :, :]

        highres_image = f.normalize(highres_image, mean=highres_image_mean, std=highres_image_std)
        lowres_image = f.normalize(lowres_image, mean=lowres_image_mean, std=lowres_image_std)

        #highres_image = f.adjust_sharpness(highres_image, 1)
        #lowres_image = f.adjust_sharpness(lowres_image, 1)

        #highres_image = f.adjust_contrast(highres_image, 1)
        #lowres_image = f.adjust_contrast(lowres_image, 1)

        #highres_image = f.adjust_brightness(highres_image, 1)
        #lowres_image = f.adjust_brightness(lowres_image, 1)

        print(highres_image.dtype)
        print(lowres_image.dtype)


        return {
            'highres': highres_image,
            'lowres': lowres_image
        }
