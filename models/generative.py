import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Upsample(scale_factor=2) # nearest neighbor instead
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, h):
        if (len(x.shape) != 4):
            raise Exception("Input tensor is incorrect shape")

        x = self.up(x)
        if h is not None:
            x = torch.cat([x, h], dim=1) # attach on channel
        return self.conv(x)

class Generative(nn.Module):
    def __init__(self, image_size, rgb = True):
        super(Generative, self).__init__()
        if (image_size % 4 != 0):
            raise Exception("Image must be a factor of 8")
        filters1 = 32
        filters2 = 64 
        padding = 1 # to ensure convolutions dont change size

        # size x / 2
        if rgb:
            self.d1 = Down(3, filters1) # rgb channels 
        else:
            self.d1 = Down(1, filters1) # rgb channels

        # size x / 4
        self.d2 = Down(filters1, filters2)

        bilinear = True
        self.u2 = Up(filters2 + filters1, filters1, bilinear=bilinear)
        if rgb:
            self.u3 = Up(filters1, filters2, bilinear=bilinear) # rgb output
        else:
            self.u3 = Up(filters1, filters2, bilinear=bilinear) # rgb output
        if rgb:
            self.last = nn.Conv2d(filters2, 3, kernel_size=3, padding=1)
        else:
            self.last = nn.Conv2d(filters2, 1, kernel_size=3, padding=1)

        

    def forward(self, x):
        # size x / 2
        x1 = self.d1(x)
        # size x / 4
        x2 = self.d2(x1)

        # size x / 2
        y = self.u2(x2, x1) # upsample y and concat with x1
        # size x
        y = self.u3(y, None)
        y = self.last(y)
        return y # image should be same size as x
