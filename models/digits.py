import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import timm

class DigitModel(nn.Module):
    """
    Digit Classifier model
    """
    def __init__(self, ):
        super(DigitModel, self).__init__()

        self.m = timm.create_model('resnet18', in_chans = 1, pretrained=False, num_classes=10, global_pool='avg')
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        """
        x = batch of input images NCHW
        """
        return self.m(x)