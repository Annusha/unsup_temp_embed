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
    opt.dataset_root = '/media/data/kukleva/lab/Breakfast'


    data_subfolder = ['kinetics', 'data', 's1', 'video'][opt.data_type]
    opt.data = os.path.join(opt.dataset_root, 'feat', data_subfolder)

    opt.gt = os.path.join(opt.data, opt.gt)

    opt.ext = ['gz', 'gz', 'txt', 'avi'][opt.data_type]
    opt.feature_dim = [400, 64, 64, 0][opt.data_type]

    opt.bg = False

    if opt.all:
        opt.subaction = 'all'

    update_opt_str()

    logger = path_logger()

    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        logger.debug('%s: %s' % (arg, getattr(opt, arg)))

