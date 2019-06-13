#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'March 2019'

import os

from ute.utils.arg_pars import opt
from ute.utils.util_functions import update_opt_str
from ute.utils.logging_setup import path_logger


def update():
    opt.dataset_root = './dummy_data'

    opt.data = os.path.join(opt.dataset_root, 'features')
    opt.subfolder = ''  # can be useful in case of different features for the same dataset

    opt.gt = os.path.join(opt.dataset_root, opt.gt)

    opt.ext = ['txt'][opt.data_type]
    opt.feature_dim = [64][opt.data_type]
    opt.embed_dim = 20  # dimensionality of the desirable embedding

    # %s is the name of the subaction
    # the name of the model to resume
    opt.resume_str = '%s_test'

    update_opt_str()

    logger = path_logger()

    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        logger.debug('%s: %s' % (arg, getattr(opt, arg)))