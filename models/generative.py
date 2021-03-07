import torch
import torch.nn as nn
import torch.nn.Functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
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
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
        
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class DownSampleBlock(nn.Module):
     def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in, out_channels),
            nn.Conv2d(out, out),
            nn.ReLU,
            nn.BatchNorm(out),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.layers(x)

class UpSampleBlock(nn.Module):
    
    def forward(self, x, h):
        self.layers(x) + h


class Generative(nn.Module):
    def __init__(self, image_size):
        super(Generative, self).__init__()
        if (image_size % 4 != 0):
            raise Exception("Image must be a factor of 4")
        # in_channels, num filters, kernal_size
        filters1 = 20   #32
        filters2 = 20   #64
        filters3 = 20   #128
        filters4 = 20   #256
        filters5 = 20   #128
        filters6 = 20   #64
        filters7 = 10
        padding = 1 # to ensure convolutions dont change size
        # FIX THIS: delete n
        n = image_size/4 #each image is 1 feature at this point dependent on image_size
        self.c1 = nn.Conv2d(3, filters1, 3, padding=padding) #convolution for color image
        self.c2 = nn.Conv2d(filters1, filters2, 3, padding=padding)
        self.m1 = nn.MaxPool2d(2) # downsample by 2
        self.c3 = nn.Conv2d(filters2, filters3, 3, padding = padding)
        self.c4 = nn.Conv2d(filters3, filters4, 3, padding = padding)
        self.m2 = nn.MaxPool2d(2) # downsample by 2
        self.c5 = nn.Conv2d(filters4, filters5, 3, padding=padding)
        self.c6 = nn.Conv2d(filters5, filters6, 3, padding=padding)
        self.m3 = nn.MaxPool2d(2) # downsample by 2
        self.c7 = nn.Conv2d(filters6, filters7, 1) # 1d conv to reduce filter size

        # self.u1 = nn.MaxUnpool2d(n) # suggestion: Bilinear upsampling
        self.d1 = nn.ConvTranspose2d(filters7, filters5, 3, padding=padding)
        self.d2 = nn.ConvTranspose2d(filters5, filters4, 3, padding=padding)
        # self.u2 = nn.MaxUnpool2d(2)
        self.d3 = nn.ConvTranspose2d(filters4, filters3, 3, padding=padding)
        self.d4 = nn.ConvTranspose2d(filters3, filters2, 3, padding=padding)
        # self.u3 = nn.MaxUnpool2d(2)
        self.d5 = nn.ConvTranspose2d(filters2, filters1, 3, padding=padding)
        self.d6 = nn.ConvTranspose2d(filters1, 1, 3, padding=padding)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.c2(x1)
        x2 = self.m1(x1)
        x2 = self.c3(x2)
        x2 = self.c4(x2)
        x3 = self.m2(x2)
        x3 = self.c5(x3)
        x3 = self.c6(x3)

        #x4 = 1xFilters
        x4 = self.m3(x3)
        x4 = self.c7(x4)

        y = self.u1(x4) + x3
        y = self.d1(y)
        y = self.d2(y)
        y = self.u2(y) + x2
        y = self.d3(y) 
        y = self.d4(y) 
        y = self.u3(y) + x1
        y = self.d5(y)
        y = self.d6(y)
        return y



        
