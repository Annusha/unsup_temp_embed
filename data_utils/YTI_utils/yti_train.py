#!/usr/bin/env python

"""Implementation and improvement of the paper:
Unsupervised learning and segmentation of complex activities from video.
"""

__author__ = 'Anna Kukleva'
__date__ = 'June 2019'

import sys
import os
sys.path.append(os.path.abspath('.').split('data_utils')[0])

from ute.utils.arg_pars import opt
from data_utils.YTI_utils.update_argpars import update
from ute.ute_pipeline import temp_embed, all_actions


if __name__ == '__main__':

    # set root
    opt.dataset_root = '/sequoia/data1/akukleva/projects/unsup_temp_embed/yti'

    # set activity
    # 'changing_tire', 'coffee', 'jump_car', 'cpr', 'repot'
    # all
    opt.subaction = 'coffee'
    # set feature extension and dimensionality
    opt.ext = 'txt'
    opt.feature_dim = 3000

    # model name can be 'mlp' or 'nothing' for no embedding (just raw features)
    opt.model_name = 'mlp'

    # do not load model
    opt.load_model = False

    # use background noise (e.g. for YTI)
    opt.bg = True
    opt.bg_trh = 75

    # update log name and absolute paths
    update()

    # run temporal embedding
    if opt.subaction == 'all':
        actions = ['changing_tire', 'coffee', 'jump_car', 'cpr', 'repot']
        all_actions(actions)
    else:
        temp_embed()


