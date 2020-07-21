"""Implementation and improvement of the paper:
Unsupervised learning and segmentation of complex activities from video.
"""

__author__ = 'Anna Kukleva'
__date__ = 'June 2019'

from ute.corpus import Corpus
from ute.utils.arg_pars import opt
from ute.utils.logging_setup import logger
from ute.utils.util_functions import timing, update_opt_str, join_return_stat, parse_return_stat


@timing
def temp_embed():
    corpus = Corpus(subaction=opt.subaction) # loads all videos, features, and gt

    logger.debug('Corpus with poses created')
    if opt.model_name in ['mlp']:
        # trains or loads a new model and uses it to extracxt temporal embeddings for each video
        corpus.regression_training()
    if opt.model_name == 'nothing':
        corpus.without_temp_emed()

    corpus.clustering()
    corpus.gaussian_model()

    corpus.accuracy_corpus()

    if opt.resume_segmentation:
        corpus.resume_segmentation()
    else:
        corpus.viterbi_decoding()

    corpus.accuracy_corpus('final')

    return corpus.return_stat


@timing
def all_actions(actions):
    return_stat_all = None
    lr_init = opt.lr
    for action in actions:
        opt.subaction = action
        if not opt.resume:
            opt.lr = lr_init
        update_opt_str()
        return_stat_single = temp_embed()
        return_stat_all = join_return_stat(return_stat_all, return_stat_single)
    logger.debug(return_stat_all)
    parse_return_stat(return_stat_all)


def resume_segmentation(iterations=10):
    logger.debug('Resume segmentation')
    corpus = Corpus(subaction=opt.action)

    for iteration in range(iterations):
        logger.debug('Iteration %d' % iteration)
        corpus.iter = iteration
        corpus.resume_segmentation()
        corpus.accuracy_corpus()
    corpus.accuracy_corpus()