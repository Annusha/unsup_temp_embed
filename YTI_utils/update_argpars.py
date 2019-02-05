#!/usr/bin/env python

"""Update parameters which directly depends on the dataset.
"""

__author__ = 'Anna Kukleva'
__date__ = 'Deceber 2018'

import os

from utils.arg_pars import opt
from utils.util_functions import update_opt_str
from utils.logging_setup import path_logger


def update():
    opt.dataset_root = '/media/data/kukleva/lab/YTInstructions'
    opt.data = os.path.join(opt.dataset_root, 'VISION_txt')
    opt.gt = os.path.join(opt.data, opt.gt)


    opt.ext = 'txt'
    opt.feature_dim = 3000

    opt.bg = True
    opt.high = False

    update_opt_str()

    logger = path_logger()

    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        logger.debug('%s: %s' % (arg, getattr(opt, arg)))


