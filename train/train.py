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
from data.dataloader import get_lfw_datasets
import numpy as np
import tabulate

"""
Training script for AdvFaceGAN
"""

# Hyperparameters
isLFW = True # Labelled faces in the wild or image net
EPOCHS = 1
BATCHSIZE = 8
INPUT_SIZE = 28
TARGET = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET = [TARGET] * BATCHSIZE
TARGET = torch.tensor(TARGET).long().to(device)
CLAMP = 1
percent_gx = .07
alternate = 4

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

    for i, img in enumerate(tqdm(train_loader)):

        img = img.to(device)
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        generator, discriminator, digits = models
        is_generator = True
        if i % alternate == 0:
            # train adversarial Generator
            generator.train()
            discriminator.eval()
        else:
            # train discriminator
            generator.eval()
            discriminator.train()
            is_generator = False
        digits.eval()

        Gx = percent_gx * generator.forward(img) # generate perturbation

        adv = Gx + img # perturbed instance
        predx = discriminator.forward(img) # discrimnator prediction on instance x
        predp = discriminator.forward(adv) # discriminator prediction on perturbed x
        predt = digits.forward(adv) # model prediction on perturbed x

        loss = criterion(predx, predp, Gx, predt, fool_class, is_generator)

        #backprop
        train_loss.append(loss.item())
        predt = predt.argmax(keepdim = False, dim = 1)
        train_acc.append((predt == TARGET).sum().cpu().detach().numpy() / BATCHSIZE)
        loss.backward()

        nn.utils.clip_grad_norm_(generator.parameters(), 1, norm_type=2.0)
        nn.utils.clip_grad_norm_(discriminator.parameters(), 1, norm_type=2.0)
        if (i % alternate == 0):
            optimizer_g.step()
        else:
            optimizer_d.step()

    return np.array(train_loss).mean(), np.array(train_acc).mean()


def validate(models, valid_loader, fool_class, criterion):

    valid_loss = []
    valid_acc = []

    for i, img in enumerate(tqdm(valid_loader)):

        img = img.to(device)

        generator, discriminator, digits = models

        generator.eval()
        discriminator.eval()
        digits.eval()

        Gx = percent_gx * generator.forward(img) # generate perturbation

        adv = Gx + img # perturbed instance
        predx = discriminator.forward(img) # discrimnator prediction on instance x
        predp = discriminator.forward(adv) # discriminator prediction on perturbed x
        predt = digits.forward(adv) # model prediction on perturbed x
        is_generator = True
        if i % alternate != 0:
            is_generator = False
        loss = criterion(predx, predp, Gx, predt, fool_class, is_generator)

        #backprop
        valid_loss.append(loss.item())
        predt = predt.argmax(keepdim = False, dim = 1)
        valid_acc.append((predt == TARGET).sum().cpu().detach().numpy() / BATCHSIZE)

    return np.array(valid_loss).mean(), np.array(valid_acc).mean()

def add_models():
    g_m = Generative(INPUT_SIZE, rgb = False).to(device)
    d_m = Discriminative(INPUT_SIZE, rgb = False).to(device)
    digits = DigitModel().to(device)
    digits.load_state_dict(torch.load("../data/models/digits/digits_best.pth", map_location=device))
    digits.eval()
    return g_m, d_m, digits

def dataset(LFW=True):
    if (LFW):
        # LFW
        train_set, valid_set = get_lfw_datasets("../data/lfw-deepfunneled")
    else:
        #ImageNet
        train_path = "../data/imagenet-mini/train"
        train_set, valid_set, _ = get_image_net_datasets(train_path)
    return train_set, valid_set


def main():

    g_m, d_m, digits = add_models()

    train_set, valid_set = dataset(isLFW)

    train_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True, num_workers=4, drop_last = True)
    valid_loader = DataLoader(valid_set, batch_size=BATCHSIZE, shuffle=False, num_workers=1, drop_last = True)

    criterion = Combined_loss(alpha=.4, beta=.2, c=.1, device=device)

    optimizer_g = optim.Adam(
        list(g_m.parameters()),
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
            f = open("./train_log.txt", 'w')
        else:
            f = open("./train_log.txt", 'a')

        f.write("Avg. Train Loss: %3f\n" % np.array(train_loss).mean())
        f.write("Avg. Train Acc: %3f\n" % np.array(train_acc).mean())

        valid_loss, valid_acc = validate([g_m, d_m, digits], valid_loader, TARGET, criterion)
        f.write("Avg. Valid Loss: %3f\n" % np.array(valid_loss).mean())
        f.write("Avg. Valid Acc: %3f\n" % np.array(valid_acc).mean())
        print(valid_loss, valid_acc)
        if (valid_acc > best_valid):
            f.write("New best!\n")
            torch.save(g_m.state_dict(), "../data/models/generator.pth")
            torch.save(d_m.state_dict(), "../data/models/discriminator.pth")
            best_valid = valid_acc

        f.close()

        scheduler_g.step()
        scheduler_d.step()




if __name__ == "__main__":
    try:
        main()
        print("training complete!")
    except Exception as e:
        print("Encountered Exception - message below:")
        traceback.print_exc()
        