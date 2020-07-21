#!/usr/bin/env python

"""Inherited corpus, since we don't use ground truth labels to separate videos
into true action classes but into sets given by clusterization on the higher
level in hierarchy.
"""

__author__ = 'Anna Kukleva'
__date__ = 'July 2020'


import numpy as np
import torch

from ute.corpus import Corpus as GroundTruthCorpus
from ute.utils.arg_pars import opt
from ute.utils.logging_setup import logger


class CorpusWrapper(GroundTruthCorpus):
    def __init__(self, videos, features, K, embedding=None):
        # todo: define bunch of parameters for the corpus
        subaction = ''
        logger.debug('%s' % subaction)

        super().__init__(K=K,
                         subaction=subaction)

        self._videos = list(np.array(videos).copy())
        self._features = features.copy()
        self._embedding = embedding

        self._update_fg_mask()

    def _init_videos(self):
        logger.debug('nothing should happen')

    def pipeline(self, iterations=1, epochs=30, dim=20, lr=1e-3):
        opt.epochs = epochs
        opt.resume = False
        opt.embed_dim = dim
        opt.lr = lr
        assert self._embedding is not None
        self._embedded_feat = torch.Tensor(self._features)
        self._embedded_feat = self._embedding.embedded(self._embedded_feat).detach().numpy()

        self.clustering()

        for iteration in range(iterations):
            logger.debug('Iteration %d' % iteration)
            self.iter = iteration

            self.clustering()
            self.gaussian_model()

            self.accuracy_corpus()

            self.viterbi_decoding()

        self.accuracy_corpus('final')

    def pr_gt(self, pr_idx_start):
        long_pr = []
        long_gt = []

        for video in self._videos:
            long_pr += list(video._z)
            long_gt += list(video.gt)

        if opt.bg:
            long_pr = map(lambda x: x + pr_idx_start if x != -1 else x, long_pr)
        else:
            long_pr = np.asarray(long_pr) + pr_idx_start

        return list(long_pr), long_gt

    def stat(self):
        return self.return_stat




