# FIX THIS: rename however you want
from models.generative import Down
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminative(nn.Module):
    def __init__(self, image_size, rgb = True):
        super(Discriminative, self).__init__()
        if (image_size % 4 != 0):
            raise Exception("Image must be a factor of 8")
        # in_channels, num filters, kernal_size
        filters1 = 32
        filters2 = 64
        filters3 = 128
        padding = 1 # to ensure convolutions dont change size

        # size x / 2
        if (rgb):
            self.d1 = Down(3, filters1) # rgb channels 
        else:
            self.d1 = Down(1, filters1) # rgb channels
        
        # size x / 4
        self.d2 = Down(filters1, filters2)

        # size x / 8
        self.d3 = Down(filters2, filters3)

        self.gap = torch.nn.AdaptiveAvgPool2d((1,1))

        hidden = 100
        self.L = nn.Linear(filters2, hidden)
        self.L2 = nn.Linear(hidden, 1)



    def forward(self, x):
        # size x / 2
        x1 = self.d1(x)
        # size x / 4
        x2 = self.d2(x1)
        # size x / 8
        # x3 = self.d3(x2)

        flatten = self.gap(x2)
        bs, c, h, w = flatten.shape
        flatten = flatten.reshape((bs, -1))
        y = self.L(flatten)
        y = F.relu(y)
        y = self.L2(y)
        return y

        

        
