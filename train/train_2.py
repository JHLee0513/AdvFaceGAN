import sys, os
import traceback
from tqdm import tqdm
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
# from torchvision.datasets import MNIST

from models.generative import Generative
from models.discriminative import Discriminative
from models.digits import DigitModel

from loss.loss import Combined_loss
from data.dataloader import get_datasets, LFWDataset
import cv2
import numpy as np
import tabulate
from sklearn.model_selection import train_test_split
from pathlib import Path

"""
Training script for AdvFaceGAN
"""

EPOCHS = 1000
BATCHSIZE = 2
INPUT_SIZE = 28
TARGET = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET = [TARGET] * BATCHSIZE
TARGET = torch.tensor(TARGET).long().to(device)
CLAMP = 0.1

def train(models, optimizer_g, optimizer_d, train_loader, fool_class, criterion):
    """
    Train one epoch of AdvFaceGAN

    models - [generator, discriminator, arcface]
    optimizer - optimizer
    train_loader - data loader
    fool_class - number for class trying to fool    
    """

    train_loss = []
    train_acc = []

    for i, (img,labels) in enumerate(tqdm(train_loader)):

        img = img.to(device)
        labels = labels.to(device)
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        generator, discriminator, digits = models

        if i % 2 == 0:
            # train adversarial Generator
            generator.train()
            discriminator.eval()
        else:
            discriminator.train()
        digits.eval()

        Gx = generator.forward(img) # generate perturbation
        Gx = torch.clamp(Gx, -CLAMP, CLAMP)

        adv = Gx + img # perturbed instance
        predx = discriminator.forward(img) # discrimnator prediction on instance x
        predp = discriminator.forward(adv) # discriminator prediction on perturbed x
        predt = digits.forward(adv) # model prediction on perturbed x

        loss = criterion(predx, predp, Gx, predt, fool_class, i)

        #backprop
        train_loss.append(loss.item())
        predt = predt.argmax(keepdim = False, dim = 1)
        train_acc.append((predt == TARGET).sum().cpu().detach().numpy() / BATCHSIZE)
        loss.backward()

        nn.utils.clip_grad_norm_(generator.parameters(), 1, norm_type=2.0)
        nn.utils.clip_grad_norm_(discriminator.parameters(), 1, norm_type=2.0)
        if (i % 2 == 0):
            optimizer_g.step()
        else:
            optimizer_d.step()

    return np.array(train_loss).mean(), np.array(train_acc).mean()
    # print("\nAvg. Train Loss: %3f\n" % np.array(train_loss).mean())


def validate(models, valid_loader, fool_class, criterion):

    valid_loss = []
    valid_acc = []

    for i, (img,labels) in enumerate(tqdm(valid_loader)):

        img = img.to(device)
        labels = labels.to(device)

        generator, discriminator, digits = models

        generator.eval()
        discriminator.eval()
        digits.eval()

        Gx = generator.forward(img) # generate perturbation
        Gx = torch.clamp(Gx, -CLAMP, CLAMP)

        adv = Gx + img # perturbed instance
        predx = discriminator.forward(img) # discrimnator prediction on instance x
        predp = discriminator.forward(adv) # discriminator prediction on perturbed x
        predt = digits.forward(adv) # model prediction on perturbed x

        loss = criterion(predx, predp, Gx, predt, fool_class, i)

        #backprop
        valid_loss.append(loss.item())
        predt = predt.argmax(keepdim = False, dim = 1)
        valid_acc.append((predt == TARGET).sum().cpu().detach().numpy() / BATCHSIZE)

    return np.array(valid_loss).mean(), np.array(valid_acc).mean()

def get_lfw_datasets(img_path):

    faces = []

    for path in Path(img_path).rglob('*.jpg'):
        faces.append(path)

    faces = faces[:4]

    # print(faces)

    train, valid = train_test_split(faces, test_size = 0.5, random_state = 42)

    # print(train, valid)

    train_set = LFWDataset(train, type = 'train')
    valid_set = LFWDataset(valid, type = 'valid')

    return train_set, valid_set

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g_m = Generative(INPUT_SIZE, rgb = False).to(device)
    d_m = Discriminative(INPUT_SIZE, rgb = False).to(device)
    digits = DigitModel().to(device)
    digits.load_state_dict(torch.load("/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/train/digits/digits_best.pth"))
    digits.eval()

    # mnist
    # train_set, val_set = get_datasets("../data/train.csv")
    # LFW
    
    train_set, val_set = get_lfw_datasets("../data/lfw/lfw-deepfunneled/lfw-deepfunneled")

    train_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True, num_workers=6, drop_last = True)
    valid_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=False, num_workers=6, drop_last = True)

    criterion = Combined_loss()

    optimizer_g = optim.Adam(
        list(g_m.parameters()),# + list(ac.parameters()),
        lr=0.001)

    optimizer_d = optim.Adam(
        list(d_m.parameters()),
        lr=0.001)

    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, EPOCHS, eta_min=0)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, EPOCHS, eta_min=0)

    best_valid = 0.

    for epoch in range(EPOCHS):
        train_loss, train_acc = train([g_m, d_m, digits], optimizer_g, optimizer_d, train_loader, TARGET, criterion)

        if epoch == 0:
            f = open("./train_log_2.txt", 'w')
        else:
            f = open("./train_log_2.txt", 'a')

        f.write("Avg. Train Loss: %3f\n" % np.array(train_loss).mean())
        f.write("Avg. Train Acc: %3f\n" % np.array(train_acc).mean())
        # print("\n Avg. Train Loss: %3f\n" % np.array(train_loss).mean())

        valid_loss, valid_acc = validate([g_m, d_m, digits], valid_loader, TARGET, criterion)
        f.write("Avg. Valid Loss: %3f\n" % np.array(valid_loss).mean())
        f.write("Avg. Valid Acc: %3f\n" % np.array(valid_acc).mean())
        
        if (valid_acc > best_valid):
            f.write("New best!\n")
            torch.save(g_m.state_dict(), "./generator_2.pth")
            torch.save(d_m.state_dict(), "./discriminator_2.pth")
            best_valid = valid_acc

        f.close()
        
        scheduler_g.step()
        scheduler_d.step()



if __name__ == "__main__":

    # test on dummy data

    # dummy = cv2.imread("/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/InsightFace_Pytorch/data/facebank/5/343.jpg")
    # dummy = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
    # dummy = cv2.resize(dummy, (INPUT_SIZE, INPUT_SIZE))
    # ac = FaceModel().to(device)
    # ac.eval()
    # target, value = ac.infer([dummy])
    # print(target, value)

    try:
        main()
    except Exception as e:
        print("Encountered Exception - message below:")
        traceback.print_exc()
    finally:
        print("training complete!")