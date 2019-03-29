#!/usr/bin/env python

"""Baseline for relative time embedding: learn regression model in terms of
relative time.
"""

__author__ = 'Anna Kukleva'
__date__ = 'September 2018'

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.arg_pars import opt
from utils.logging_setup import logger


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        ### concat
        self.fc1 = nn.Linear(opt.feature_dim * opt.concat + opt.rt_concat + opt.label,
                             opt.embed_dim * 2)

        # self.fc1 = nn.Linear(opt.feature_dim, opt.embed_dim * 2)
        self.fc2 = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.fc_last = nn.Linear(opt.embed_dim, 1)
        self._init_weights()

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        # x = F.sigmoid(self.fc_last(x))
        x = self.fc_last(x)
        # return x.view(-1)
        return x

    def embedded(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def create_model():
    torch.manual_seed(opt.seed)
    model = MLP().cuda()
    loss = nn.MSELoss(reduction='sum').cuda()
    # loss = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    logger.debug(str(model))
    logger.debug(str(loss))
    logger.debug(str(optimizer))
    return model, loss, optimizer

