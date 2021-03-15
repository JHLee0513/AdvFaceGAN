import sys, os
import traceback
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision import transforms

from models.generative import Generative
from models.discriminative import Discriminative
from models.digits import DigitModel
from loss.loss import Combined_loss

import matplotlib.pyplot as plt

# sys.path.append(os.path.abspath(os.path.join('..', 'InsightFace_Pytorch')))

from PIL import Image
import numpy as np
import tabulate
from os import listdir
from os.path import isfile
"""
Test on any image
"""

INPUT_SIZE = 28
TARGET = 4 # target to fool classifier as
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_paths(samples, TOTAL_PATH):
    folder = listdir(TOTAL_PATH)
    for str_ in folder:
        if (isfile(TOTAL_PATH+str_)):
            samples.append(TOTAL_PATH + str_)


if __name__ == "__main__":

    # test on dummy data

    samples = [
        "../data/samples/me_28x28.jpg",
    ]
    PATH = "../facebank/"

    add_paths(samples, PATH + "brad_pitt/")
    add_paths(samples, PATH + "banana/")
    add_paths(samples, PATH + "sunflower/")
    
    for name in samples:
        GENERATOR_PATH = "../weights2/generator_2.pth" # change to path of prefered generator
        dummy = Image.open(name)
        gen = Generative(INPUT_SIZE, rgb = False).to(device)
        # map_location for sad cpu gang
        gen.load_state_dict(torch.load(GENERATOR_PATH, map_location=device))
        
        test_transform = trans.Compose([
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor()
            ])
        
        dummy_ten = test_transform(dummy).to(device).unsqueeze(0)

        orig = dummy_ten
        print(orig.max())
        gx = gen(dummy_ten)
        adv = gx + dummy_ten
        plotimg = torch.cat((orig.detach().reshape(28, 28), gx.detach().reshape(28, 28), adv.detach().reshape(28, 28)), 1) 
        plt.imshow(plotimg.detach(), cmap='gray')
        plt.show()

        ac = DigitModel().to(device)
        ac.load_state_dict(torch.load("../weights/digits/digits_best.pth", map_location=device))
        ac.eval()
        target = ac(adv)
        print(target.argmax(keepdim=False, dim=-1))