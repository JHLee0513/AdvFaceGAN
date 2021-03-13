import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Networks losses for adversarial generator GAN is as follows:
1. GAN loss
3. Adversarial loss
4. norm (hinge) loss
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gan_loss(nn.Module):
    """
    Discriminative loss + generative loss
    Assumes label 1 is for real and 0 for fake
    """
    def __init__(self):
        super(Gan_loss, self).__init__()
        # self.l = nn.BCEWithLogitsLoss(reduction = 'mean')
        self.l = nn.MSELoss(reduction = 'mean')

    def forward(self, input_real, input_fake, itr):
        assert (input_real.shape == input_fake.shape)
        bs = input_real.shape[0]

        pos = torch.ones([bs, 1]).to(device)
        neg = torch.zeros([bs, 1]).to(device)

        # dis_loss = self.l(input_real, pos) + self.l(input_fake, neg)
        dis_loss = self.l(input_real, torch.ones_like(input_real, device=device))
        dis_loss = dis_loss + self.l(input_fake, torch.zeros_like(input_real, device=device))
        # dis_loss = torch.log(torch.sigmoid(input_real))
        gen_loss = self.l(input_fake, torch.ones_like(input_fake, device=device))
        # print("DIS LOSS: %3f \t GEN LOSS: %3f" % (dis_loss.mean().item(), gen_loss.mean().item()))
        # loss = dis_loss + gen_loss
        # loss = loss.mean()
        if (itr % 2 == 0):
            # generator
            return gen_loss.mean()
        
        # discriminator
        return dis_loss.mean() / 2
        # return loss.mean()

class Adv_loss(nn.Module):
    """
    Adversarial loss
    """
    def __init__(self):
        super(Adv_loss, self).__init__()
        self.l = nn.NLLLoss(reduction = 'mean')

    def forward(self, input, target):
        """
        input: model prediction OHE (N, C)
        target: target class (N)
        """
        # return F.cross_entropy(input, target, reduction = "mean")
        # return self.l(torch.log(input + 1e-10), target)
        # print(input[0], target[0])
        # return F.mse_loss(input, target, reduction = 'sum')
        return F.mse_loss(input, target, reduction = 'sum')

class Hinge_loss(nn.Module):
    def __init__(self, c):
        super(Hinge_loss, self).__init__()
        self.c = c

    def forward(self, input):
        """
        input: image NCHW
        """
        return torch.max(torch.tensor([0]).cuda(), torch.norm(input) - self.c)

class Combined_loss(nn.Module):
    def __init__(self, alpha = 0.1, beta = 1, c = 200):
        super(Combined_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gan_loss = Gan_loss()
        self.adv_loss = Adv_loss()
        self.hinge_loss = Hinge_loss(c)
    
    def forward(self, y, yp, Gx, t_pred, t_gt, itr):
        """
        y: discriminator prediction on origianl instance x
        yp: dicriminator prediction on perturbed instances x + G(x)
        Gx: Generated perturbance G(x)
        t_pred: prediction on perturbed  instance F(x + G(x))
        t_gt: target class (for targeted attack)

        itr: flag to notify the function which part of GAN the loss is for,
            even(including 0) - generator
            odd - discriminator
        """

        a = self.adv_loss(t_pred, t_gt)
        b = self.alpha * self.gan_loss(y, yp, itr)
        # c = self.beta * self.hinge_loss(Gx)
        # print("ADV LOSS: %3f \t GAN LOSS: %3f \t HINGE LOSS: %3f" %(a.item(), b.item(), c.item()))
        # print("ADV LOSS: %3f \t GAN LOSS: %3f \t HINGE LOSS: %3f" %(a.item(), b.item(), 0))

        if itr % 2 == 0:
            # generator
            return a + b #+ c
        return b

        # return a + b + c