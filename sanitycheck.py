import sys, os
import traceback
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans

from models.generative import Generative
from models.discriminative import Discriminative
# from models.arcface import FaceModel
from loss.loss import Combined_loss

sys.path.append(os.path.abspath(os.path.join('..', 'InsightFace_Pytorch')))
# import InsightFace_pytorch
from InsightFace_Pytorch.model import MobileFaceNet, Arcface, Backbone
from InsightFace_Pytorch.data.data_pipe import get_train_dataset
from train import FaceModel

import cv2
import numpy as np
import tabulate
"""
Training script for AdvFaceGAN
"""

EPOCHS = 75
BATCHSIZE = 24
INPUT_SIZE = 112
# TARGET = 4 # target to fool classifier as
NUM_CLASSES = 85742
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOOL_CLASS = torch.zeros((BATCHSIZE)).long().to(device) # TODO: support multi-class targeted attack

def step(model, batch, criterion):
    predictions = model(batch)
    loss = criterion(predictions)
    return predictions, loss

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

        if i > 100: 
            break

        img = img.to(device)
        labels = labels.to(device)
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        generator, discriminator, arcface = models

        if i % 2 == 0:
            # train adversarial Generator
            generator.train()
            discriminator.eval()
        else:
            # train discriminator
            generator.eval()
            discriminator.train()
        arcface.eval()


        Gx = generator.forward(img) # generate perturbation
        adv = Gx + img # perturbed instance
        predx = discriminator.forward(img) # discrimnator prediction on instance x
        predp = discriminator.forward(adv) # discriminator prediction on perturbed x
        predt, _ = arcface.forward(adv) # model prediction on perturbed x

        # generate OHE for NLL loss
        bs = predt.shape[0]
        predt_ohe = torch.zeros(bs, NUM_CLASSES).to(device)
        predt_ohe[:, predt] = 1

        loss = criterion(predx, predp, Gx, predt_ohe, fool_class, i)
        # print(loss.item())
        #backprop
        train_loss.append(loss.item())

        # print((predt == 52900).sum() / bs)

        train_acc.append((predt.cpu().detach().numpy() == 52900).sum() / bs)

        # print(train_loss)
        loss.backward()

        # print("BATCH %d" % i)
        # print(module_grad_stats(generator))
        # print("==============================================")
        # print(module_grad_stats(discriminator))

        nn.utils.clip_grad_norm_(generator.parameters(), 1, norm_type=2.0)
        nn.utils.clip_grad_norm_(discriminator.parameters(), 1, norm_type=2.0)
        # clip_grad_norm_(generator.parameters(), 1, norm_type=2.0)
        if (i % 2 == 0):
            optimizer_g.step()
        else:
            optimizer_d.step()

    return np.array(train_loss).mean(), np.array(train_acc).mean()
    # print("\nAvg. Train Loss: %3f\n" % np.array(train_loss).mean())

def validate(model, valid_loader, fool_class):
    pass
    # valid_loss = []
    # model.eval()
    # for data in valid_loader:
    #     pred, loss = step(model, data, criterion)
    #     valid_loss += [loss]

    # return np.array(valid_loss).mean()

def main():


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g_m = Generative(INPUT_SIZE).to(device)
    d_m = Discriminative(INPUT_SIZE).to(device)
    ac = FaceModel().to(device)
    ac.eval()

    ds, class_num = get_train_dataset("/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/InsightFace_Pytorch/data/faces_emore/imgs")

    train_dataset = ds
    valid_dataset = ds

    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4)

    criterion = Combined_loss()

    optimizer_g = optim.Adam(
        list(g_m.parameters()) + list(ac.parameters()),
        lr=0.01)

    optimizer_d = optim.Adam(
        list(d_m.parameters()),
        lr=0.01)
    
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, EPOCHS, eta_min=0)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, EPOCHS, eta_min=0)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train([g_m, d_m, ac], optimizer_g, optimizer_d, train_loader, FOOL_CLASS, criterion)

        if epoch == 0:
            f = open("./train_log.txt", 'w')
        else:
            f = open("./train_log.txt", 'a')

        f.write("Avg. Train Loss: %3f\n" % np.array(train_loss).mean())
        f.write("Avg. Train acc: %3f\n" % np.array(train_acc).mean())
        f.close()
        # print("\n Avg. Train Loss: %3f\n" % np.array(train_loss).mean())

        valid_loss = validate([g_m, d_m, ac], valid_loader, FOOL_CLASS)
        scheduler_g.step()
        scheduler_d.step()

    torch.save(g_m.state_dict(), "./generator.pth")
    torch.save(d_m.state_dict(), "./discriminator.pth")

if __name__ == "__main__":

    # test on dummy data
    # dummy = cv2.imread("/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/InsightFace_Pytorch/data/facebank/6/385.jpg")

    facebank_path = "/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/InsightFace_Pytorch/data/facebank"
    names = np.load(os.path.join(facebank_path, 'names.npy'))
    embeddings = torch.load(os.path.join(facebank_path,'facebank.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FOOL_CLASS = 52900
    FOOL_CLASS = embeddings[FOOL_CLASS].to(device)

    samples = [
        "/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/InsightFace_Pytorch/data/facebank/6/385.jpg",
        "/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/InsightFace_Pytorch/data/facebank/5/343.jpg",
        "/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/InsightFace_Pytorch/data/facebank/2/210.jpg",
    ]
    
    for name in samples:

        dummy = cv2.imread(name)
        dummy = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
        dummy = cv2.resize(dummy, (INPUT_SIZE, INPUT_SIZE))
        
        gen = Generative(INPUT_SIZE).to(device)
        gen.load_state_dict(torch.load("/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/generator.pth"))
        
        test_transform = trans.Compose([
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        
        dummy_ten = test_transform(dummy).to(device).unsqueeze(0)

        orig = dummy_ten
        adv = gen(dummy_ten) + dummy_ten

        ac = FaceModel().to(device)
        ac.eval()
        # target, value = ac.infer([dummy])
        target, value = ac.forward_embed(adv)
        target_o, value_o = ac.forward_embed(orig)
        print(target, value)
        print(target_o, value_o)

        src_embed = ac.forward(adv)
        src_embed_o = ac.forward(orig)
        print((src_embed - FOOL_CLASS).mean())
