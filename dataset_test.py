from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import read_image
from torchvision import datasets, transforms
from dataset import cellDataset
import torch
import matplotlib.pyplot as plt

transformCells = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToPILImage(), transforms.ToTensor()])
dataset = cellDataset("data/train.csv", transform=transformCells)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

batch = next(iter(dataloader))
images, labels, index = batch
plt.imshow(images[0,0,:,:], cmap="gray")
plt.show