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

# changed to double conv then maxpool
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
        # if (x.shape[2] != h.shape[2] or x.shape[3] != h.shape[3]):
        #     raise Exception("shapes do not match for x and h!!!")

        x = self.up(x)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        # diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if h is not None:
            x = torch.cat([x, h], dim=1) # attach on channel
        return self.conv(x)

class Generative(nn.Module):
    def __init__(self, image_size, rgb = True):
        super(Generative, self).__init__()
        if (image_size % 8 != 0):
            raise Exception("Image must be a factor of 8")
        # in_channels, num filters, kernal_size
        filters1 = 32   #32
        filters2 = 64   #64
        filters3 = 128   #128
        # filters4 = 256   #256
        padding = 1 # to ensure convolutions dont change size
        # FIX THIS: delete n
        n = image_size/4 #each image is 1 feature at this point dependent on image_size

        # size x / 2
        if rgb:
            self.d1 = Down(3, filters1) # rgb channels 
        else:
            self.d1 = Down(1, filters1) # rgb channels

        # size x / 4
        self.d2 = Down(filters1, filters2)

        # size x / 8
        self.d3 = Down(filters2, filters3)
        # self.onedim = nn.Conv2d(filters3, filters4, 1) # 1d conv to reduce filter size
        bilinear = True
        self.u1 = Up(filters3 + filters2, filters2, bilinear=bilinear)
        self.u2 = Up(filters2 + filters1, filters1, bilinear=bilinear)
        if rgb:
            self.u3 = Up(filters1, 3, bilinear=bilinear) # rgb output
        else:
            self.u3 = Up(filters1, 1, bilinear=bilinear) # grayscale output
        

    def forward(self, x):
        # size x / 2
        x1 = self.d1(x)
        # size x / 4
        x2 = self.d2(x1)
        # size x / 8
        x3 = self.d3(x2)

        # size x / 4
        y = self.u1(x3, x2) # upsample x3 and and concat with x2
        # size x / 2
        y = self.u2(y, x1) # upsample y and concat with x1
        # size x
        y = self.u3(y, None)
        return y # image should be same size as x

