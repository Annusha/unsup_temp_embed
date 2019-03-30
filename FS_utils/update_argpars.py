#!/usr/bin/env python

"""Update parameters which directly depends on the dataset.
"""

__author__ = 'Anna Kukleva'
__date__ = 'November 2018'

import os

from utils.arg_pars import opt
from utils.util_functions import update_opt_str
from utils.logging_setup import path_logger


def update():
    opt.dataset_root = '/media/data/kukleva/lab/50salads'

    data_subfolder = ['i3d', '', 's1'][opt.data_type]
    opt.data = os.path.join(opt.dataset_root, 'features', data_subfolder)

    opt.gt = os.path.join(opt.data, opt.gt)

    opt.ext = ['npy', '', 'txt'][opt.data_type]
    opt.feature_dim = [2048, 0, 64][opt.data_type]
    opt.embed_dim = 30

    opt.bg = False

    if opt.all:
        opt.subaction = 'all'

    update_opt_str()

    logger = path_logger()

    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        logger.debug('%s: %s' % (arg, getattr(opt, arg)))

