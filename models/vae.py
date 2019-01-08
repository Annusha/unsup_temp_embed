#!/usr/bin/env python

"""Using vae representation for features in concern of relative time.
Implementation inherited from pytorch examples code.
"""

__author__ = 'Anna Kukleva'
__date__ = 'September 2018'

import torch.nn as nn
import torch
from torch.nn import functional as F

from utils.arg_pars import opt, logger


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc_1z = nn.Linear(opt.feature_dim + opt.vae_dim, opt.embed_dim * 2)
        self.fc_2z_mu = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.fc_2z_var = nn.Linear(opt.embed_dim * 2, opt.embed_dim)

        self.fc_z2 = nn.Linear(opt.embed_dim, opt.embed_dim * 2)
        # self.fc_z1 = nn.Linear(opt.embed_dim * 2, opt.feature_dim + opt.vae_dim)
        self.fc_z1_feat = nn.Linear(opt.embed_dim * 2, opt.feature_dim)
        self.fc_z1_time = nn.Linear(opt.embed_dim * 2, opt.vae_dim)

        self._init_weights()

    def encode(self, x):
        h_1z = F.relu(self.fc_1z(x))
        return self.fc_2z_mu(h_1z), self.fc_2z_var(h_1z)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        # h_z2 = F.relu(self.fc_z2(z))
        # return self.fc_z1(h_z2)
        h_z2 = F.relu(self.fc_z2(z))
        h_z1_feat = self.fc_z1_feat(h_z2)
        h_z1_time = F.sigmoid(self.fc_z1_time(h_z2))
        return torch.cat((h_z1_feat, h_z1_time), 1)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, opt.feature_dim + opt.vae_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def embedded(self, x):
        x, _ = self.encode(x.view(-1, opt.feature_dim + opt.vae_dim))
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def loss_function_vae(recon_x, x, mu, logvar):
    # MSE = F.mse_loss(recon_x, x.view(-1, opt.feature_dim + opt.vae_dim), size_average=False)
    x = x.view(-1, opt.feature_dim + opt.vae_dim)
    MSE = F.smooth_l1_loss(recon_x[:, :opt.feature_dim],
                           x[:, :opt.feature_dim],
                           size_average=False)
    BCE = F.binary_cross_entropy(recon_x[:, opt.feature_dim:],
                                 x[:, opt.feature_dim:],
                                 size_average=False)
    # MSE = F.smooth_l1_loss(recon_x, x.view(-1, opt.vae_dim), size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # KLD = -10.0 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD + BCE

def create_model():
    torch.manual_seed(opt.seed)
    model = VAE().cuda()
    loss = loss_function_vae
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    logger.debug(str(model))
    logger.debug(str(loss))
    logger.debug(str(optimizer))
    return model, loss, optimizer



