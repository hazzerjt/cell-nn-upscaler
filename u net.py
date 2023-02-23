import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from network_components import *
from dataset import cellDataset


transformCells = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
dataset = datasets.ImageFolder('data\Cells', transformCells)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

#dataset = cellDataset(annotations_file="data/Cells/labels2.csv", img_dir="data/Cells/8bit images")
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

torch.set_grad_enabled(True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.BatchNorm2d1 = nn.BatchNorm2d(32)
        self.reLU1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.BatchNorm2d2 = nn.BatchNorm2d(64)
        self.reLU2 = nn.ReLU(inplace=True)

        self.down1 = down(64, 96, 128, 3)
        self.down2 = down(128, 192, 256, 3)
        self.down3 = down(256, 384, 512, 3)
        self.down4 = down(512, 768, 1024, 3)

        self.up1 = up(1024, 512, 3)
        self.up2 = up(512, 256, 3)
        self.up3 = up(256, 128, 3)
        self.up4 = up(128, 64, 3)
        self.up5 = up(64, 32, 3)

        self.conv3 = nn.Conv2d(32, 1, 1)#Reduces the number of channels to 1

    def forward(self, t):
        t = t
        t = self.conv1(t)
        t = self.BatchNorm2d1(t)
        t = self.reLU1(t)
        t = self.conv2(t)
        t = self.BatchNorm2d2(t)
        t = self.reLU2(t)
        print(t.size())

        t = self.down1(t)
        print(t.size())
        t = self.down2(t)
        print(t.size())
        t = self.down3(t)
        print(t.size())
        t = self.down4(t)
        print(t.size())

        t = self.up1(t)
        print(t.size())
        t = self.up2(t)
        print(t.size())
        t = self.up3(t)
        print(t.size())
        t = self.up4(t)
        print(t.size())
        t = self.up5(t)
        print(t.size())

        t = self.conv3(t)
        print(t.size())

        return t

    def __repr__(self):
        return "Bunny Kitten"

network = Network()
batch = next(iter(dataloader))
images, labels = batch

plt.imshow(images[0,0,:,:], cmap="gray")
plt.show()

preds = network(images)

plt.imshow(preds[0,0,:,:].detach().numpy(), cmap="gray")
plt.show()