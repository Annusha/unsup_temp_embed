#!/usr/bin/env python

"""Grid search funtions where can be defined and evaluated numerous of
parameters.
"""

__author__ = 'Anna Kukleva'
__date__ = 'October 2018'

from utils.arg_pars import opt
from utils.logging_setup import logger
from utils.util_functions import update_opt_str


actions = ['coffee', 'cereals', 'milk', 'tea', 'juice', 'sandwich', 'salat', 'friedegg', 'scrambledegg', 'pancake']


# def grid_search(**kwargs):
#     n_params = len(kwargs)
#     f = kwargs['f']
#     del kwargs['f']
#     keys = list(kwargs.keys())
#
#     line = '\n\nSET: '
#     for key, val in kwargs.items():
#         setattr(opt, key, val)
#         if key == 'lr':
#             subline = '%s: %.1e\t'
#         elif isinstance(val, float):
#             subline = '%s: %.3f\t'
#         elif isinstance(val, int):
#             subline = '%s: %d\t'
#         else:  # assume everything else
#             val = str(val)
#             subline = '%s: %s\t'
#         subline = subline % (key, val)
#
#     for arg in vars(opt):
#         logger.debug('%s: %s' % (arg, getattr(opt, arg)))


def grid_search(f):
    # for epoch, dim, lr in grid:
    # grid = [[30, 20, 1e-3],
    #         [30, 30, 1e-3],
    #         [30, 40, 1e-3]]

    # radius = [1.0, 1.5, 2.0]
    # epochs = [5, 10, 20]
    # dims = [20, 50, 100, 200]


    # for r in radius:
    #     for epoch in epochs:
    #         for dim in dims:
    #         opt.bg_trh = r
    # logger.debug('\n\nSET: radius: %.1e  dim: %d  epochs: %d\n' %
    # (r, dim, epoch))
    # weights = [10.0, 20.0]
    # bg_trh = [1, 1.5, 2]

    # concats = [3, 9]
    # dims = [40, 80]
    # epochs = [30, 60]
    #
    # for concat in concats:
    #     for epoch in epochs:
    #         for dim in dims:

    # grid = [[40, 90, 1e-2],
    #         [20, 90, 1e-4],
    #         [30, 90, 1e-4],
    #         [40, 90, 1e-4]]

    epochs = [10, 20, 40]
    dims = [20, 30, 40]
    lrs = [1e-2, 1e-3, 1e-4]
    norms = [False, True]


    # resume_template = 'grid.vit._%s_mlp_!pose_full_vae1_time10.0_epochs%d_embed%d_n2_ordering_gmm1_one_!gt_lr%s_lr_zeros_b0_v1_l0_c1_'
    # resume_template = 'fixed.order._%s_mlp_!pose_full_vae0_time10.0_epochs%d_embed%d_n1_!ordering_gmm1_one_!gt_lr%s_lr_zeros_b0_v1_l0_c1_'

    # for dim, epoch, lr in grid:
    for norm in norms:
        for epoch in epochs:
            for lr in lrs:
                for dim in dims:

                    opt.embed_dim = dim
                    opt.epochs = epoch
                    # opt.concat = concat

                    opt.lr = lr
                    opt.f_norm = norm
                    # opt.time_weight = w
                    # opt.bg_trh = w
                    # opt.resume_str = resume_template % (opt.subaction, epoch, dim, str(lr))

                    logger.debug('\n\nSET: dim: %d  epochs: %d, lr: %.1e, norm: %s\n' %
                                 (dim, epoch, lr, str(norm)))
                    update_opt_str()
                    for arg in vars(opt):
                        logger.debug('%s: %s' % (arg, getattr(opt, arg)))
                    f()


def grid_search_tcn(f):
    dims = [20, 40, 80]
    lrs = [1e-3, 1e-4, 1e-5]
    dropouts = [0.2, 0.5]
    epochs = [10, 20, 40]
    levels = [1, 2]
    ksizes = [3, 4, 5]

    counter = 0

    for epoch in epochs:
        for dropout in dropouts:
            for level in levels:
                for ksize in ksizes:
                    for lr in lrs:
                        for dim in dims:
                            opt.embed_dim = dim
                            opt.lr = lr
                            opt.ksize = ksize
                            opt.levels = level
                            opt.dropout = dropout
                            opt.epochs = epoch

                            logger.debug('\n\n%d SET: dim: %d lr: %.1e ksize %d levels %d dropout %.1f epochs %d\n'
                                         % (counter, dim, lr, ksize, level, dropout, epoch))
                            update_opt_str()
                            f()

                            counter += 1


def grid_search_seeds(f, start):
    for idx, i in enumerate(range(start, start+5)):
        opt.seed = i
        logger.debug('\n\n%d SET: seed: %d\n' %(idx, i))
        update_opt_str()
        f()
