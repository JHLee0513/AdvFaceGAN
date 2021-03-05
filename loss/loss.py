# import torch
import torch.nn as nn
import torch.nn.Functional as F

"""
Networks losses for adversarial generator GAN is as follows:
1. GAN loss
3. Adversarial loss
4. norm (hinge) loss
"""

class gan_loss(nn.Module):
    """
    Discriminative loss + generative loss
    Assumes label 1 is for real and 0 for fake
    """
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(gan_loss, self).__init__()

    def forward(self, input_real: Tensor, input_fake: Tensor, ) -> Tensor:
        dis_loss = torch.log(torch.sigmoid(input_real))
        gen_loss = torch.log(1 - torch.sigmoid(input_fake))
        loss = dis_loss + gen_loss
        if self.reduce == 'mean':
            return loss.mean()
        elif self.reduce == "sum":
            return loss.sum()
        return loss

class adv_loss(nn.Module):
    """
    Adversarial loss
    """
    def __init__(self, size_average=None, reduce=None, reduction : str = 'mean') -> None:
        super(adv_loss, self).__init__()

    def forward(self, input, target):
        """
        input: model prediction
        target: target OHE vector of class
        """
        return F.cross_entropy(intput, target, reduction = self.reduction)
    

class hinge_loss(nn.Module):
    def __init__(c, reduction: str = 'mean') -> None:
        super(hinge_loss).__init__()
        self.c = c

    def forward(self, input):
        """
        input: image NCHW
        """
        return torch.max(0, torch.norm(input) - c)


class combined_loss(nn.Module):
    def __init__(alpha = 1, beta = 1, c = 1) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gan_loss = gan_loss()
        self.adv_loss = adv_loss()
        self.hinge_loss = hinge_loss(c)
    
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