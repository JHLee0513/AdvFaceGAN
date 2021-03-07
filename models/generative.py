import torch
import torch.nn as nn
import torch.nn.Functional as F

class Generative(nn.Module):
    def __init__(self, image_size):
        super(Generative, self).__init__()
        if (image_size % 4 != 0):
            raise Exception("Image must be a factor of 4")
        # in_channels, num filters, kernal_size
        filters1 = 20
        filters2 = 20
        filters3 = 20
        filters4 = 20
        filters5 = 20
        filters6 = 20
        filters7 = 10
        padding = 1 # to ensure convolutions dont change size
        n = image_size/4 #each image is 1 feature at this point dependent on image_size
        self.c1 = nn.Conv2d(3, filters1, 3, padding=padding) #convolution for color image
        self.c2 = nn.Conv2d(filters1, filters2, 3, padding=padding)
        self.m1 = nn.MaxPool2d(2) # downsample by 2
        self.c3 = nn.Conv2d(filters2, filters3, 3, padding = padding)
        self.c4 = nn.Conv2d(filters3, filters4, 3, padding = padding)
        self.m2 = nn.MaxPool2d(2) # downsample by 2
        self.c5 = nn.Conv2d(filters4, filters5, 3, padding=padding)
        self.c6 = nn.Conv2d(filters5, filters6, 3, padding=padding)
        self.m3 = nn.MaxPool2d(n) # downsample by 2
        self.c7 = nn.Conv2d(filters6, filters7, 1) # 1d conv to reduce filter size

        self.u1 = nn.MaxUnpool2d(n)
        self.d1 = nn.ConvTranspose2d(filters7, filters5, 3, padding=padding)
        self.d2 = nn.ConvTranspose2d(filters5, filters4, 3, padding=padding)
        self.u2 = nn.MaxUnpool2d(2)
        self.d3 = nn.ConvTranspose2d(filters4, filters3, 3, padding=padding)
        self.d4 = nn.ConvTranspose2d(filters3, filters2, 3, padding=padding)
        self.u3 = nn.MaxUnpool2d(2)
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



        
