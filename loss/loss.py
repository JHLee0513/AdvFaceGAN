import torch
import torch.nn as nn
import torch.nn.functional as F

class Gan_loss(nn.Module):
    """
    Discriminative loss + generative loss
    Assumes label 1 is for real and 0 for fake
    """
    def __init__(self, device):
        super(Gan_loss, self).__init__()
        self.l = nn.BCEWithLogitsLoss(reduction = 'mean')
        self.device = device

    def forward(self, input_real, input_fake, is_generator):
        assert (input_real.shape == input_fake.shape)
        bs = input_real.shape[0]

        pos = torch.ones([bs, 1]).to(self.device)
        neg = torch.zeros([bs, 1]).to(self.device)

        dis_loss = self.l(input_real, pos) + self.l(input_fake, neg)
        gen_loss = self.l(input_fake, pos)
        if (is_generator):
            # generator
            return gen_loss.mean()
        
        # discriminator
        return dis_loss.mean() / 2

class Adv_loss(nn.Module):
    """
    Adversarial loss
    """
    def __init__(self):
        super(Adv_loss, self).__init__()

    def forward(self, input, target):
        """
        input: model prediction OHE (N, C)
        target: target class (N)
        """
        return F.cross_entropy(input, target, reduction = "mean")

class Hinge_loss(nn.Module):
    def __init__(self, c, device):
        super(Hinge_loss, self).__init__()
        self.c = c
        self.device = device

    def forward(self, input):
        """
        input: image NCHW
        """
        return torch.max(torch.tensor([0]).to(self.device), torch.norm(input) - self.c)

class Combined_loss(nn.Module):
    def __init__(self, alpha = 0.1, beta = 1, c = 100, device="cpu"):
        super(Combined_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gan_loss = Gan_loss(device)
        self.adv_loss = Adv_loss()
        self.hinge_loss = Hinge_loss(c, device)
    
    def forward(self, y, yp, Gx, t_pred, t_gt, is_generator):
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
        b = self.alpha * self.gan_loss(y, yp, is_generator)
        c = self.beta * self.hinge_loss(Gx)

        if is_generator:
            # generator
            return a + b + c
        return b
