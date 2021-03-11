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

import cv2
import numpy as np

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


class FaceModel(nn.Module):
    def __init__(self, ):
        super(FaceModel, self).__init__()

        # self.backbone = MobileFaceNet(512)
        # w = torch.load("/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/models/model_mobilefacenet.pth")

        self.backbone = Backbone(50, 0.6, mode='ir_se')
        w = torch.load("/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/InsightFace_Pytorch/work_space/models/model_ir_se50.pth")

        self.backbone.load_state_dict(w)
        # self.ac = Arcface(embedding_size=512)

        self.test_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        facebank_path = "/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/InsightFace_Pytorch/data/facebank"

        self.embeddings = torch.load(os.path.join(facebank_path,'facebank.pth'))
        self.names = np.load(os.path.join(facebank_path, 'names.npy'))
        self.threshold = 1.5

    def forward(self, x): #, label):
        """
        x = batch of input images NCHW
        """
        source_embs = self.backbone(x)
        
        diff = source_embs.unsqueeze(-1) - self.embeddings.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum
    
    def infer(self, faces):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        target_embs = self.embeddings
        embs = []
        for img in faces:
            embs.append(self.backbone(self.test_transform(img).to(self.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum    

def step(model, batch, criterion):
    predictions = model(batch)
    loss = criterion(predictions)
    return predictions, loss

def train(models, optimizer, train_loader, fool_class, criterion):
    """
    Train one epoch of AdvFaceGAN

    models - [generator, discriminator, arcface]
    optimizer - optimizer
    train_loader - data loader
    fool_class - number for class trying to fool
    """
    train_loss = []

    for i, (img,labels) in enumerate(tqdm(train_loader)):

        if i > 100: 
            break

        img = img.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

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
        loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), 1, norm_type=2.0)
        nn.utils.clip_grad_norm_(discriminator.parameters(), 1, norm_type=2.0)
        # clip_grad_norm_(generator.parameters(), 1, norm_type=2.0)
        optimizer.step()

    return np.array(train_loss).mean()
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

    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4)

    criterion = Combined_loss()

    optimizer = optim.Adam(
        list(g_m.parameters()) + list(d_m.parameters()) + list(ac.parameters()),
        lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=0)

    for epoch in range(EPOCHS):
        train_loss = train([g_m, d_m, ac], optimizer, train_loader, FOOL_CLASS, criterion)
        print("\n Avg. Train Loss: %3f\n" % np.array(train_loss))

        valid_loss = validate([g_m, d_m, ac], valid_loader, FOOL_CLASS)
        scheduler.step()

    torch.save(g_m.state_dict(), "./generator.pth")
    torch.save(d_m.state_dict(), "./discriminator.pth")

if __name__ == "__main__":

    # test on dummy data
    # dummy = cv2.imread("/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/InsightFace_Pytorch/data/facebank/0/1.jpg")
    # dummy = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
    # dummy = cv2.resize(dummy, (INPUT_SIZE, INPUT_SIZE))
    # target, value = ac.infer([dummy])

    try:
        main()
    except Exception as e:
        print("Encountered Exception - message below:")
        traceback.print_exc()
    finally:
        print("training complete!")
