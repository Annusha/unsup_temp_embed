#!/usr/bin/env python

"""Implementation of training and testing functions for embedding."""

__all__ = ['training', 'load_model']
__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import torch
import torch.backends.cudnn as cudnn
from os.path import join
import time
import numpy as np
import random

from ute.utils.arg_pars import opt
from ute.utils.logging_setup import logger
from ute.utils.util_functions import Averaging, adjust_lr
from ute.utils.util_functions import dir_check


def training(train_loader, epochs, save, **kwargs):
    """Training pipeline for embedding.

    Args:
        train_loader: iterator within dataset
        epochs: how much training epochs to perform
        n_subact: number of subactions in current complex activity
        mnist: if training with mnist dataset (just to test everything how well
            it works)
    Returns:
        trained pytorch model
    """
    logger.debug('create model')

    # make everything deterministic -> seed setup
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    model = kwargs['model']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']

    cudnn.benchmark = True

    batch_time = Averaging()
    data_time = Averaging()
    losses = Averaging()

    adjustable_lr = opt.lr

    logger.debug('epochs: %s', epochs)
    for epoch in range(epochs):
        # model.cuda()
        model.to(opt.device)
        model.train()

        logger.debug('Epoch # %d' % epoch)
        if opt.lr_adj:
            # if epoch in [int(epochs * 0.3), int(epochs * 0.7)]:
            # if epoch in [int(epochs * 0.5)]:
            if epoch % 30 == 0 and epoch > 0:
                adjustable_lr = adjust_lr(optimizer, adjustable_lr)
                logger.debug('lr: %f' % adjustable_lr)
        end = time.time()
        for i, (features, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)
            features = features.float()
            labels = labels.float().to(opt.device)
            if opt.device == 'cuda':
                features = features.cuda(non_blocking=True)
            # features = features.float().cuda(non_blocking=True)
            # labels = labels.float().cuda()
            output = model(features)
            loss_values = loss(output.squeeze(), labels.squeeze())
            losses.update(loss_values.item(), features.size(0))

            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0 and i:
                logger.debug('Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
        logger.debug('loss: %f' % losses.avg)
        losses.reset()

    opt.resume_str = join(opt.dataset_root, 'models',
                          '%s.pth.tar' % opt.log_str)
    if save:
        save_dict = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        if opt.global_pipe:
            dir_check(join(opt.dataset_root, 'models', 'global'))
            opt.resume_str = join(opt.dataset_root, 'models', 'global',
                                  '%s.pth.tar' % opt.log_str)
        else:
            dir_check(join(opt.dataset_root, 'models'))
        torch.save(save_dict, opt.resume_str)
    return model


def load_model():
    if opt.loaded_model_name:
        if opt.global_pipe:
            resume_str = opt.loaded_model_name
        else:
            resume_str = opt.loaded_model_name % opt.subaction
        # resume_str = opt.resume_str
    else:
        resume_str = opt.log_str + '.pth.tar'
    # opt.loaded_model_name = resume_str
    if opt.device == 'cpu':
        checkpoint = torch.load(join(opt.dataset_root, 'models',
                                     '%s' % resume_str),
                                map_location='cpu')
    else:
        checkpoint = torch.load(join(opt.dataset_root, 'models',
                                 '%s' % resume_str))
    checkpoint = checkpoint['state_dict']
    logger.debug('loaded model: ' + '%s' % resume_str)
    return checkpoint

