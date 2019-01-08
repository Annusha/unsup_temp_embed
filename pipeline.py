#!/usr/bin/env python

"""Implementation and improvement of the paper:
Unsupervised learning and segmentation of complex activities from video.
"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

from corpus import Corpus
from utils.arg_pars import opt
from utils.logging_setup import logger
from utils.util_functions import timing, update_opt_str, join_return_stat, parse_return_stat
from utils.grid_search import grid_search_tcn as gs
import BF_utils.update_argpars as bf_utils
import YTI_utils.update_argpars as yti_utils


@timing
def temp_embed(iterations=1):
    corpus = Corpus(Q=opt.gmm,
                    subaction=opt.subaction)

    logger.debug('Corpus with poses created')
    if opt.model_name in ['mlp', 'tcn']:
        corpus.regression_training()
    if opt.model_name == 'nothing':
        corpus.without_temp_emed()

    if opt.gaussian_cl:
        corpus.gaussian_clustering()
    else:
        corpus.clustering()

    for iteration in range(iterations):
        logger.debug('Iteration %d' % iteration)
        corpus.iter = iteration

        if not opt.gaussian_cl:
            corpus.gaussian_model()

        corpus.accuracy_corpus()

        if opt.resume_segmentation:
            corpus.resume_segmentation()
        else:
            if opt.viterbi:
                # corpus.viterbi_decoding()
                # corpus.accuracy_corpus(prefix='pure vit ')

                # corpus.viterbi_ordering()
                corpus.ordering_sampler()
                corpus.rho_sampling()
                # corpus.accuracy_corpus(prefix='vit+ord ')


                corpus.viterbi_decoding()
                # corpus.viterbi_alex_decoding()
            else:
                corpus.subactivity_sampler()

                corpus.ordering_sampler()
                corpus.rho_sampling()
    corpus.accuracy_corpus('final')

    return corpus.return_stat

@timing
def all_actions():
    return_stat_all = None
    if opt.dataset == 'bf':
        actions = ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'friedegg', 'salat', 'pancake']
    if opt.dataset == 'yti':
        actions = ['changing_tire', 'coffee', 'jump_car', 'cpr', 'repot']
    lr_init = opt.lr
    for action in actions:
        opt.subaction = action
        if not opt.resume:
            opt.lr = lr_init
        update_opt_str()
        if opt.viterbi:
            return_stat_single = temp_embed(iterations=1)
        else:
            return_stat_single = temp_embed(iterations=5)
        return_stat_all = join_return_stat(return_stat_all, return_stat_single)
    parse_return_stat(return_stat_all)


@timing
def grid_search():
    # f = temp_embed(iterations=1)
    gs(temp_embed)


def resume_segmentation(iterations=10):
    logger.debug('Resume segmentation')
    corpus = Corpus(Q=opt.gmm,
                    subaction=opt.subaction)

    for iteration in range(iterations):
        logger.debug('Iteration %d' % iteration)
        corpus.iter = iteration
        corpus.resume_segmentation()
        corpus.accuracy_corpus()
    corpus.accuracy_corpus()


if __name__ == '__main__':
    if opt.dataset == 'bf':
        bf_utils.update()
    if opt.dataset == 'yti':
        yti_utils.update()
    if opt.all:
        all_actions()
    elif opt.grid_search:
        grid_search()
    else:
        temp_embed()
