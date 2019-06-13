#!/usr/bin/env python

"""Compare precomputed gaussians. One of them is fitted on the entire video
collection other one is one the entire excluded one video."""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import numpy as np
import os

from ute.utils.arg_pars import opt, logger

n1 = '00'
n2_0 = '0%d'
n2 = '01'
# one many

for i in range(167):
    n2 = n2_0%i
    mean_one = np.loadtxt(os.path.join(opt.data, 'gauss_pr', 'mean_%s_gmm_%d_#n_%s' % ('one', 0, n1)))
    mean_many = np.loadtxt(os.path.join(opt.data, 'gauss_pr', 'mean_%s_gmm_%d_#n_%s' % ('many', 0, n2)))

    var_one = np.loadtxt(os.path.join(opt.data, 'gauss_pr', 'var_%s_gmm_%d_#n_%s' % ('one', 0, n1)))
    var_many = np.loadtxt(os.path.join(opt.data, 'gauss_pr', 'var_%s_gmm_%d_#n_%s' % ('many', 0, n2)))

    mse = np.mean((mean_one - mean_many) ** 2)
    abs_diff = np.mean(np.abs(mean_one - mean_many))
    max_diff_mse = np.max((mean_one - mean_many) ** 2)
    max_diff_abs = np.max(np.abs(mean_one - mean_many))

    logger.debug('\nmean:\nmse: %f\nmean abs_diff: %f\nmax_diff_mse: %f\nmax_diff_abs: %f'
                 '\nrange first: (%f, %f)\nrange second: (%f, %f)\n' %
                 (mse, abs_diff, max_diff_mse, max_diff_abs,
                  np.min(mean_one), np.max(mean_one),
                 np.min(mean_many), np.max(mean_many)))


    mse = np.mean((var_one - var_many) ** 2)
    abs_diff = np.mean(np.abs(var_one - var_many))
    max_diff_mse = np.max((var_one - var_many) ** 2)
    max_diff_abs = np.max(np.abs(var_one - var_many))

    logger.debug('\nvariance:\nmse: %f\nmean abs_diff: %f\nmax_diff_mse: %f\nmax_diff_abs: %f'
                 '\nrange first: (%f, %f), \nrange second: (%f, %f)' %
                 (mse, abs_diff, max_diff_mse, max_diff_abs,
                 np.min(var_one), np.max(var_one),
                 np.min(var_many), np.max(var_many)))

logger.debug('end')
