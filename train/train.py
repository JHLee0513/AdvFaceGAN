import torch
import torch.nn as nn
import torch.nn.Functional as F
from models import Generative
from loss import Combined_loss


"""
g_M - Generative model
arcface - nn for arcface
fool_class - number for class trying to fool
"""
def train_one_epoch(train_set, g_m, arcface, fool_class, f_loss):
    arcface.eval() # freeze network
    opt = torch.optim.SGD(g_m.parameters(), lr=.01, momentum=.9)

    for batch in train_set:
        g_m.zero_grad()
        Gx = g_m.forward(batch) # generate adv filter
        adv = Gx + batch
        one_hot_face = arcface(adv)
        gen_loss = Combined_loss(?, ?, Gx, one_hot_face, fool_class)
        
        #backprop
        gen_loss.backward()
        opt.step()

def train(epochs, train_set, g_m, arcface, fool_class, f_loss):
    for epoch in epochs:
        print("Epoch: " + str(epoch))
        train_one_epoch(train_set, g_m, arcface, fool_class, f_loss)

if __name__ == "__main__":
    image_size = 64
    f_loss = Combined_loss()
    g_m = Generative(image_size)

    train(10, None, g_m, None, 0, f_loss) 
