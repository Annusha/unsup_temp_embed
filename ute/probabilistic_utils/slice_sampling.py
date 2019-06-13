#!/usr/bin/env python

"""Slice sampling only for one dim functions."""

__all__ = ['slice_sampling']
__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import numpy as np

from ute.probabilistic_utils.mallow import Mallow


def step_out(x_init, u_prime, logpdf, w=5):
    r = np.random.uniform(0, 1)
    x_l = x_init - r * w
    x_r = x_init + (1 - r) * w
    while logpdf(x_l) > u_prime and abs(x_l - x_r) < 1e+3:
        x_l -= w
    while logpdf(x_r) > u_prime and abs(x_l - x_r) < 1e+3:
        x_r += w
    return x_l, x_r


def shrinking(x_prime, x_init, x_l, x_r):
    if x_prime > x_init:
        x_r = x_prime
    else:
        x_l = x_prime
    return x_l, x_r


def slice_sample(x_init, logpdf):
    eval_pdf = logpdf(x_init)
    assert eval_pdf >= 0
    u_prime = np.random.uniform(0, eval_pdf)
    x_l, x_r = step_out(x_init, u_prime, logpdf)
    # print('stepout ', x_l, x_r, x_init, u_prime, eval_pdf)
    counter = 0
    while True:
        counter += 1
        # print(x_l, x_r)
        x_prime = np.random.uniform(x_l, x_r)
        eval_pdf = logpdf(x_prime)
        if eval_pdf > u_prime:
            return x_prime
        else:
            x_l, x_r = shrinking(x_prime, x_init, x_l, x_r)
        if abs(x_l - x_r) < 1e-3 or counter > 1e+3:
            return None


def slice_sampling(burnin, x_init, logpdf):
    for _ in range(burnin):
        x_hat = slice_sample(x_init, logpdf)
        while x_hat is None:
            x_hat = slice_sample(x_init, logpdf)
        x_init = x_hat
    return x_init

if __name__ == '__main__':
    mal = Mallow(K=6)
    inv_pdf = lambda x: -1. / mal.logpdf(x)
    mal.set_sample_params(sum_inv_vals=0, k=1, N=167)
    x = slice_sampling(burnin=5, x_init=1, logpdf=inv_pdf)
    print(x)