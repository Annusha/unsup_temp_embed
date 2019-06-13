#!/usr/bin/env python

"""Update parameters which directly depends on the dataset.
"""

__author__ = 'Anna Kukleva'
__date__ = 'November 2018'

import os
import os.path as ops

from ute.utils.arg_pars import opt
from ute.utils.util_functions import update_opt_str, dir_check
from ute.utils.logging_setup import path_logger
import torch



def update():

    opt.data = ops.join(opt.dataset_root, 'features')
    opt.gt = ops.join(opt.dataset_root, 'groundTruth')
    opt.output_dir = ops.join(opt.dataset_root, 'output')
    opt.mapping_dir = ops.join(opt.dataset_root, 'mapping')
    dir_check(opt.output_dir)
    opt.f_norm = True
    if torch.cuda.is_available():
        opt.device = 'cuda'
    # # specify here root for your Breakfast dataset
    # opt.dataset_root = '/media/data/kukleva/lab/Breakfast'
    #
    # # specify here subfolder with
    # data_subfolder = ['kinetics', 'data', 's1', 'video', 'OPN', 'videovector', 'videovector', 'videodarwin'][opt.data_type]
    # opt.data = os.path.join(opt.dataset_root, 'feat', data_subfolder)
    #
    # opt.gt = os.path.join(opt.data, opt.gt)
    #
    # opt.ext = ['gz', 'gz', 'txt', 'avi', 'txt', 'txt', 'txt', 'txt'][opt.data_type]
    # opt.feature_dim = [400, 64, 64, 0, 1024, 4096, 64, 128][opt.data_type]
    # opt.subfolder = 'ascii'
    #
    # if opt.data_type in [5, 6]:
    #     # ascii_subsampled ascii_imgnet ascii_target ascii_context
    #     i = 0 if opt.data_type == 5 else 1
    #     opt.subfolder = ['ascii_%s', 'ascii_subsampled'][i]
    #     if i == 0:
    #         data_type = ['target', 'context'][0]
    #         resume_iter = [25, 50][1]
    #         opt.subfolder = opt.subfolder % data_type
    #         opt.prefix = opt.prefix + '.%s.%d' % (data_type, resume_iter)
    #     opt.frame_frequency = 5
    #
    # if opt.data_type in [4, 5, 6, 7]:
    #     exception_list = ['P03_webcam02_P03_tea',
    #                       'P26_webcam02_P26_cereals',
    #                       'P27_webcam02_P27_coffee',
    #                       'P34_cam01_P34_coffee',
    #                       'P51_webcam01_P51_coffee',
    #                       'P52_stereo01_P52_sandwich']
    #
    #     for root, dirs, files in os.walk(os.path.join(opt.data, opt.subfolder)):
    #         if files:
    #             for filename in files:
    #                 if 'P51_webcam01_P51_coffee' in filename:
    #                     print('P51_webcam01_P51_coffee')
    #                 for exception in exception_list:
    #                     if exception in filename:
    #                         try:
    #                             os.remove(os.path.join(root, filename))
    #                             os.remove(os.path.join(opt.gt, filename))
    #                         except FileNotFoundError:
    #                             pass
    #
    # if opt.data_type == 6:
    #     opt.model_name = 'mlp'
    #
    # if opt.data_type in [0, 4, 5, 7]:
    #     opt.model_name = 'nothing'


    opt.bg = False  # YTI argument
    opt.gr_lev = ''  # 50Salads argument
    #
    # if opt.all:
    #     opt.subaction = 'all'

    update_opt_str()

    logger = path_logger()

    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        logger.debug('%s: %s' % (arg, getattr(opt, arg)))

