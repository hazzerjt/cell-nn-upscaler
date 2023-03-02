import torch.nn as nn
import torch.nn.functional as F
import torch

class up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size, padding)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size, padding)

    def forward(self, t1, t2):
        t1 = self.up(t1)
        diffY = t2.size()[2] - t1.size()[2]
        diffX = t2.size()[3] - t1.size()[3]

        t1 = F.pad(t1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        t = torch.cat([t2, t1], dim=1)
        return self.conv(t)

class down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size, padding))

    def forward(self, t):
        return self.maxpool_conv(t)

class inc(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.inc = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True))


    def forward(self, t):
        return self.inc(t)

class final(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.up = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))

    def forward(self, t):
        return self.up(t)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, t):
        return self.double_conv(t)


