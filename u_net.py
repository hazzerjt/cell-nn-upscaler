from network_components import *

class Network(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.inc = inc(1, 32, 64, kernel_size, kernel_size // 2)

        self.down1 = down(64, 128, kernel_size, kernel_size // 2)
        self.down2 = down(128, 256, kernel_size, kernel_size // 2)
        self.down3 = down(256, 512, kernel_size, kernel_size // 2)
        self.down4 = down(512, 1024, kernel_size, kernel_size // 2)


        self.up1 = up(1024, 512, kernel_size, kernel_size // 2)
        self.up2 = up(512, 256, kernel_size, kernel_size // 2)
        self.up3 = up(256, 128, kernel_size, kernel_size // 2)
        self.up4 = up(128, 64, kernel_size, kernel_size // 2)

        self.final = final(64, 32, kernel_size, kernel_size // 2)

        self.conv3 = nn.Conv2d(32, 1, 1)#Reduces the number of channels to 1

    def forward(self, t):
        t = t
        t1 = self.inc(t)

        t2 = self.down1(t1)
        t3 = self.down2(t2)
        t4 = self.down3(t3)
        t5 = self.down4(t4)

        t = self.up1(t5, t4)
        t = self.up2(t, t3)
        t = self.up3(t, t2)
        t = self.up4(t, t1)
        t = self.final(t)
        t = self.conv3(t)

        return t

    def __repr__(self):
        return "Bunny Kitten"