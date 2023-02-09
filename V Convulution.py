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

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)

        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2))
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3)

        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2))
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3)

        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2))
        self.conv7 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3)
        self.conv8 = nn.Conv2d(in_channels=384, out_channels=768, kernel_size=3)

        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2))
        self.conv9 = nn.Conv2d(in_channels=768, out_channels=1536, kernel_size=3)
        self.conv10 = nn.Conv2d(in_channels=1536, out_channels=3072, kernel_size=3)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv11 = nn.Conv2d(in_channels=3072, out_channels=1536, kernel_size=3)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv12 = nn.Conv2d(in_channels=1536, out_channels=768, kernel_size=3)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv13 = nn.Conv2d(in_channels=768, out_channels=384, kernel_size=3)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv14 = nn.Conv2d(in_channels=384, out_channels=192, kernel_size=3)

        self.out = nn.Linear(in_features=192, out_features=96)



    def forward(self, t):
        t = t
        t = self.conv1(t)
        t = self.conv2(t)

        t = self.maxpool_conv(t)
        t = self.conv3(t)
        t = self.conv4(t)

        t = self.maxpool_conv(t)
        t = self.conv5(t)
        t = self.conv6(t)

        t = self.maxpool_conv(t)
        t = self.conv7(t)
        t = self.conv8(t)

        t = self.maxpool_conv(t)
        t = self.conv9(t)
        t = self.conv10(t)

        t = self.up(t)
        t = self.conv11(t)

        t = self.up(t)
        t = self.conv12(t)

        t = self.up(t)
        t = self.conv13(t)

        t = self.up(t)
        t = self.conv14(t)

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