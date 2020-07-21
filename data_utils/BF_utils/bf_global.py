#!/usr/bin/env python

""" Test or train global embedding on the breakfast with unknown high-level activity classes
"""

__author__ = 'Anna Kukleva'
__date__ = 'July 2020'

import sys
import os
sys.path.append(os.path.abspath('.').split('data_utils')[0])

from ute.utils.arg_pars import opt
from data_utils.BF_utils.update_argpars import update
from ute.global_corpus import run_pipeline

if __name__ == '__main__':
    opt.global_pipe = True
    opt.subaction = 'global'

    # set root
    opt.dataset_root = '/BS/kukleva/work/data/bf/fv'

    # global parameters
    opt.global_K = 5
    opt.global_k_prime = 10

    # set feature extension and dimensionality
    opt.ext = 'txt'
    opt.feature_dim = 64

    # model name can be 'mlp' or 'nothing' for no embedding (just raw features)
    opt.model_name = 'mlp'

    # load an already trained model (stored in the models directory in dataset_root)
    opt.load_model = True
    # opt.loaded_model_name = 'global.pth.tar'
    opt.loaded_model_name = 'global%d_%d.pth.tar' % (opt.global_K, opt.global_k_prime)

    # update log name and absolute paths
    update()

    run_pipeline()

