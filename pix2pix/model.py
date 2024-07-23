import torch
import torch.nn as nn
from . import networks

class Pix2Pix():
    def __init__(self, in_channels, out_channels, isTrain=True, gan_mode='lsgan', device='cuda', lr=0.0002):
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = networks.Unet(in_channels, out_channels)

        if self.isTrain:
            self.netD = networks.PatchGAN(in_channels + out_channels, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d)

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(gan_mode).to(device)
            self.criterionL1 = nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr, betas=(0.5, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def setup(self): # from basemodel
        if self.isTrain:
            self.schedulers = 