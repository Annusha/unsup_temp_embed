#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'June 2019'

import os.path as ops
import torch

from ute.utils.arg_pars import opt
from ute.utils.util_functions import update_opt_str, dir_check
from ute.utils.logging_setup import path_logger


def update():
    opt.data = ops.join(opt.dataset_root, 'features')
    opt.gt = ops.join(opt.dataset_root, 'groundTruth')
    opt.output_dir = ops.join(opt.dataset_root, 'output')
    opt.mapping_dir = ops.join(opt.dataset_root, 'mapping')
    dir_check(opt.output_dir)
    opt.f_norm = True
    if torch.cuda.is_available():
        opt.device = 'cuda'

    opt.gr_lev = ''  # 50Salads argument

    if opt.model_name == 'nothing':
        opt.load_embed_feat = True

    update_opt_str()

    logger = path_logger()

    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        logger.debug('%s: %s' % (arg, getattr(opt, arg)))