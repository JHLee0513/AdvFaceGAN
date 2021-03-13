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

sys.path.append(os.path.abspath(os.path.join('..', 'InsightFace_Pytorch')))

from PIL import Image
import numpy as np
import tabulate

"""
Test on any image
"""

INPUT_SIZE = 28
TARGET = 4 # target to fool classifier as
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # test on dummy data

    samples = [
        "/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/data/samples/me_28x28.jpg",
    ]
    
    for name in samples:

        dummy = Image.open(name)
        gen = Generative(INPUT_SIZE, rgb = False).to(device)
        gen.load_state_dict(torch.load("/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/train/generator.pth"))
        
        test_transform = trans.Compose([
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor()
            ])
        
        dummy_ten = test_transform(dummy).to(device).unsqueeze(0)

        orig = dummy_ten
        adv = gen(dummy_ten) + dummy_ten

        ac = DigitModel().to(device)
        ac.load_state_dict(torch.load("/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/train/digits_best.pth"))
        ac.eval()
        target = ac(adv)
        print(target.argmax(keepdim=False, dim=-1))