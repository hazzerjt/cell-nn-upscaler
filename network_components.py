import torch.nn as nn

class up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size)
        self.BatchNorm2d = nn.BatchNorm2d(in_channels)
        self.reLU = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.reLU = nn.ReLU(inplace=True)

    def forward(self, t):
        return self.up(t)
class down(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size)
        self.BatchNorm2d = nn.BatchNorm2d(mid_channels)
        self.reLU = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.reLU = nn.ReLU(inplace=True)

    def forward(self, t):
        return self.maxpool_conv(t)
