import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.generative import Generative
from models.discriminative import Discriminative
# from models.arcface import FaceModel
from loss.loss import Combined_loss

sys.path.append(os.path.abspath(os.path.join('..', 'InsightFace_Pytorch')))
# import InsightFace_pytorch
from InsightFace_Pytorch.model import MobileFaceNet, Arcface

import cv2

"""
Training script for AdvFaceGAN
"""

EPOCHS = 10
BATCHSIZE = 8
INPUT_SIZE = 112
TARGET = 4 # target to fool classifier as

class FaceModel(nn.Module):
    def __init__(self, ):
        super(FaceModel, self).__init__()

        self.backbone = MobileFaceNet(512)
        w = torch.load("/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/models/model_mobilefacenet.pth")
        self.backbone.load_state_dict(w)
        self.ac = Arcface(embedding_size=512)

    def forward(self, x): #, label):
        x = self.backbone(x)
        return x
        # return self.ac(x, label)

def step(model, batch, criterion):
    predictions = model(batch)
    loss = criterion(predictions)
    return predictions, loss

def train(train_loader, g_m, d_m, arcface, fool_class):
    """
    Train one epoch of AdvFaceGAN

    train_loader - data loader
    g_m - Generative model
    d_m - Discriminative model
    arcface - arcface facial recognition model
    fool_class - number for class trying to fool
    """
    train_loss = []

    # arcface.eval() # freeze network

    # for i, (img,labels) in enumerate(tqdm(train_loader)):
    #     g_m.zero_grad()
    #     Gx = g_m.forward(batch) # generate adv filter
    #     adv = Gx + x
    #     one_hot_face = arcface(adv)
    #     gen_loss = Combined_loss(?, ?, Gx, one_hot_face, fool_class)
        
    #     #backprop
    #     gen_loss.backward()
    #     opt.step()

def validate(model, valid_loader, criterion):
    valid_loss = []
    model.eval()
    for data in valid_loader:
        pred, loss = step(model, data, criterion)
        valid_loss += [loss]

    return np.array(valid_loss).mean()

def train(epochs, train_set, g_m, arcface, fool_class, f_loss):
    for epoch in epochs:
        print("Epoch: " + str(epoch))
        train_one_epoch(train_set, g_m, arcface, fool_class, f_loss)

def main():
    train_dataset = None
    train_loader = None
    valid_dataset = None
    valid_loader = None
    criterion = combined_loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, min=0)

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion)
        valid_loss = validate(model, valid_loader, criterion)
        scheduler.step()

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f_loss = Combined_loss()
    
    g_m = Generative(INPUT_SIZE).to(device)
    d_m = Discriminative(INPUT_SIZE).to(device)
    
    ac = FaceModel().to(device)

    dummy = cv2.imread("/media/joonho1804/Storage/455FINALPROJECT/AdvFaceGAN/InsightFace_Pytorch/data/faces_emore/imgs/0/1.jpg")
    dummy = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
    # dummy = cv2.resize(dummy, (INPUT_SIZE, INPUT_SIZE))
    dummy = dummy.transpose(2, 0, 1)
    dummy = torch.tensor(dummy).float().to(device).unsqueeze(0)
    dummy = dummy / 255

    dummy = torch.cat([dummy, dummy], dim = 0)

    label = torch.tensor([1, 1]).to(device).long() # Batch x 1

    pred = ac(dummy)
    m = nn.Softmax(dim=1)
    # pred = ac(dummy, label)
    print(m(pred))
    # print(torch.softmax(pred[0]).sum())


    # ds, class_num = get_train_dataset(conf.emore_folder/'imgs')
    # loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    # return loader, class_num 

    # train(10, None, g_m, None, 0, f_loss) 
