import sys, os
import traceback
from tqdm import tqdm

sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from data.dataloader import get_datasets

from models.digits import DigitModel
from loss.loss import Combined_loss

import cv2
import numpy as np
import pandas as pd

"""
Training script for the Digit recognition model
"""
EPOCHS = 100
BATCHSIZE = 64

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # using MNIST dataset subset from Kaggle
    train_set, val_set = get_datasets("../data/train.csv")

    # transform = transforms.Compose([transforms.ToTensor(), 
    #                                 transforms.Normalize((0.5,), (0.5,))])
    # dataset = MNIST('./data', train = True, transform = transform, download=True)
    # train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])

    # train_set.dataset.transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(p=0.25),
    #     transforms.RandomVerticalFlip(p=0.25),
    #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    #     transforms.RandomAffine(20, translate=0.1, scale=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])

    model = DigitModel()
    model.to(device)

    best_valid = 0.

    train_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True, num_workers=6)
    valid_loader = DataLoader(val_set, batch_size=BATCHSIZE, shuffle=False, num_workers=6)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=0)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, optimizer, train_loader, criterion, device)

        if epoch == 0:
            f = open("./digits_train_log.txt", 'w')
        else:
            f = open("./digits_train_log.txt", 'a')

        f.write("Avg. Train Loss: %3f\n" % np.array(train_loss).mean())
        f.write("Avg. Train acc: %3f\n" % np.array(train_acc).mean())

        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)
        f.write("Avg. Valid Loss: %3f\n" % np.array(valid_loss).mean())
        f.write("Avg. Valid acc: %3f\n" % np.array(valid_acc).mean())

        scheduler.step()

        if valid_acc > best_valid:
            torch.save(model.state_dict(), "./digits_best.pth")
            best_valid = valid_acc
            f.write("New best!\n")
        f.close()

def train(model, optimizer, train_loader, criterion, device):
    """
    Train one epoch of Digit recognition model

    model
    optimizer - optimizer
    train_loader - data loader
    criterion - loss
    device - torch device
    """
    train_loss = []
    train_acc = []

    for i, (img,labels) in enumerate(tqdm(train_loader)):

        img = img.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        pred = model(img)

        loss = criterion(pred, labels)
        # print(pred.shape, labels.shape)
        acc = (pred.argmax(keepdim = False, dim = 1) == labels).sum().detach().cpu().numpy() / img.shape[0]

        train_loss.append(loss.item())

        train_acc.append(acc)

        loss.backward()
        optimizer.step()

    return np.array(train_loss).mean(), np.array(train_acc).mean()

def validate(model, valid_loader, criterion, device):
    """
    """
    valid_loss = []
    valid_acc = []

    model.eval()

    for i, (img,labels) in enumerate(tqdm(valid_loader)):

        img = img.to(device)
        labels = labels.to(device)

        pred = model(img)

        loss = criterion(pred, labels)
        acc = (pred.argmax(keepdim = False, dim = 1) == labels).sum().detach().cpu().numpy() / img.shape[0]

        valid_loss.append(loss.item())
        valid_acc.append(acc)

    model.train()
    return np.array(valid_loss).mean(), np.array(valid_acc).mean()


if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        print("Encountered Exception - message below:")
        traceback.print_exc()
    finally:
        print("training complete!")