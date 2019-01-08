#!/usr/bin/env python

"""Implementation of simple tcn model
"""

__author__ = 'Anna Kukleva'
__date__ = 'December 2018'

import torch
import torch.nn as nn

from utils.arg_pars import opt
from utils.logging_setup import logger
from models.tcn_locuslab import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self._init_weights()

    def _init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        out = self.tcn(x)
        return self.linear(out[:, :, -1])

    def embedded(self, x):
        out = self.tcn(x)
        return out[:, :, -1]


def create_model():
    torch.manual_seed(opt.seed)
    channel_sizes = [opt.embed_dim] * opt.levels
    model = TCN(input_size=opt.feature_dim + opt.rt_concat,
                output_size=opt.tcn_len,
                num_channels=channel_sizes,
                kernel_size=opt.ksize,
                dropout=opt.dropout).cuda()
    loss = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    logger.debug(str(model))
    logger.debug(str(loss))
    logger.debug(str(optimizer))
    return model, loss, optimizer
