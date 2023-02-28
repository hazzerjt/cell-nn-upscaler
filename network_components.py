import torch.nn as nn
import torch.nn.functional as F
import torch

class up(nn.Module):
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

    def forward(self, t1, t2):
        #return self.up(t)
        diffY = t2.size()[2] - t1.size()[2]
        diffX = t2.size()[3] - t1.size()[3]

        t1 = F.pad(t1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        t = torch.cat([t2, t1], dim=1)
        return self.up(t)
class down(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.down = nn.Sequential(
        nn.MaxPool2d(2),
        nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))

    def forward(self, t):
        return self.down(t)

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


