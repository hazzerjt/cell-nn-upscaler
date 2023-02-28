from network_components import *

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = inc(1, 32, 64, 3, 1)

        self.down1 = down(64, 96, 128, 3, 1)
        self.down2 = down(128, 192, 256, 3, 1)
        self.down3 = down(256, 384, 512, 3, 1)
        self.down4 = down(512, 768, 1024, 3, 1)

        self.up1 = up(1024, 512, 3, 1)
        self.up2 = up(512, 256, 3, 1)
        self.up3 = up(256, 128, 3, 1)
        self.up4 = up(128, 64, 3, 1)

        self.final = final(64, 32, 3, 1)

        self.conv3 = nn.Conv2d(32, 1, 1)#Reduces the number of channels to 1

    def forward(self, t):
        t = t
        t1 = self.inc(t)
        print(t1.size())

        t2 = self.down1(t1)
        print(t2.size())
        t3 = self.down2(t2)
        print(t3.size())
        t4 = self.down3(t3)
        print(t4.size())
        t5 = self.down4(t4)
        print(t5.size())

        t = self.up1(t5, t4)
        print(t.size())
        t = self.up2(t, t3)
        print(t.size())
        t = self.up3(t, t2)
        print(t.size())
        t = self.up4(t, t1)
        print(t.size())
        t = self.final(t)
        print(t.size())

        t = self.conv3(t)
        print(t.size())

        return t

    def __repr__(self):
        return "Bunny Kitten"