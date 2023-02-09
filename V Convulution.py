import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
dataset = datasets.ImageFolder('data\Cells', transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

torch.set_grad_enabled(True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)

        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2))
        self.conv3 = nn.Conv2d(in_channels=125, out_channels=256, kernel_size=3)

        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2))
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)

        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2))
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv9 = nn.Conv2d(in_channels=126, out_channels=64, kernel_size=3)

        self.out = nn.Linear(in_features=64, out_features=32)

    def forward(self, t):
        t = t
        t = self.conv1(t)

        t = self.maxpool_conv(t)
        t = self.conv2(t)

        t = self.maxpool_conv(t)
        t = self.conv3(t)

        t = self.maxpool_conv(t)
        t = self.conv4(t)

        t = self.maxpool_conv(t)
        t = self.conv5(t)

        t = self.up(t)
        t = self.conv6(t)

        t = self.up(t)
        t = self.conv7(t)

        t = self.up(t)
        t = self.conv8(t)

        t = self.up(t)
        t = self.conv9(t)

        t = self.out(t)

        return t

    def __repr__(self):
        return "Bunny Kitten"

network = Network()

batch = next(iter(dataloader))
images, labels = batch

plt.imshow(images[0,0,:,:])
print(labels)

preds = network(images)

print(preds.argmax(dim=1))
print(labels)
print(preds.argmax(dim=1).eq(labels))
print(preds.argmax(dim=1).eq(labels).sum())

print(preds[0])
print(labels[0])