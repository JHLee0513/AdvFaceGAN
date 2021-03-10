# import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Networks losses for adversarial generator GAN is as follows:
1. GAN loss
3. Adversarial loss
4. norm (hinge) loss
"""

class Gan_loss(nn.Module):
    """
    Discriminative loss + generative loss
    Assumes label 1 is for real and 0 for fake
    """
    def __init__(self):
        super(Gan_loss, self).__init__()

    def forward(self, input_real, input_fake):
        dis_loss = torch.log(torch.sigmoid(input_real))
        gen_loss = torch.log(1 - torch.sigmoid(input_fake))
        loss = dis_loss + gen_loss
        return loss.mean()

class Adv_loss(nn.Module):
    """
    Adversarial loss
    """
    def __init__(self):
        super(Adv_loss, self).__init__()

    def forward(self, input, target):
        """
        input: model prediction (N, C)
        target: target OHE vector of class (N)
        """
        return F.cross_entropy(input, target, reduction = "mean")

class Hinge_loss(nn.Module):
    def __init__(self, c):
        super(Hinge_loss, self).__init__()
        self.c = c

    def forward(self, input):
        """
        input: image NCHW
        """
        return torch.max(0, torch.norm(input) - c)

class Combined_loss(nn.Module):
    def __init__(self, alpha = 1, beta = 1, c = 1):
        super(Combined_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gan_loss = Gan_loss()
        self.adv_loss = Adv_loss()
        self.hinge_loss = Hinge_loss(c)
    
    def forward(self, y, yp, Gx, t_pred, t_gt):
        """
        y: discriminator prediction on origianl instance x
        yp: dicriminator prediction on perturbed instances x + G(x)
        Gx: Generated perturbance G(x)
        t_pred: prediction on perturbed  instance F(x + G(x))
        t_gt: target class (for targeted attack)
        """

        a = self.adv_loss(t_pred, t_gt)
        b = self.alpha * self.gan_loss(y, yp)
        c = self.beta * self.hinge_loss(Gx)

        return a + b + c