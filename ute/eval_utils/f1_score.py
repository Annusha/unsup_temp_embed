#!/usr/bin/env python

"""F1 score for sampled frames from gt to define how well steps were recovered.
"""

__author__ = 'Anna Kukleva'
__date__ = 'November 2018'

import numpy as np

from ute.utils.logging_setup import logger


class F1Score:
    def __init__(self, K, n_videos):
        self.sampling_ratio = 15  # number of frames per segment to sample
        self.n_experiments = 50
        self._K = K  # number of predicted segments per video
        self._n_videos = n_videos
        self._eps = 1e-8

        self.gt = None
        self.gt_sampled = None
        self.pr = None
        self.pr_sampled = None
        self.gt2pr = None
        self.mask = None
        self.exclude = []

        self.bound_masks = []  # list of masks for each segment

        self.f1_scores = []
        self._return = {}
        self._n_true_seg_all = 0

    def set_gt(self, gt):
        self.gt = np.asarray(gt)
        self.mask = np.zeros(self.gt.shape, dtype=bool)

    def set_pr(self, pr):
        self.pr = np.asarray(pr)

    def set_gt2pr(self, gt2pr):
        self.gt2pr = gt2pr

    def set_exclude(self, label):
        self.bound_masks = []
        self.exclude.append(label)
        mask_exclude = self.gt != label
        self.gt = self.gt[mask_exclude]
        self.pr = self.pr[mask_exclude]
        self.mask = np.zeros(self.gt.shape, dtype=bool)

    def _finish_init(self):
        if self.gt is not None and \
                self.pr is not None and \
                self.gt2pr is not None:
            self._pr2gt_convert()
            self._set_boundaries()

    def _pr2gt_convert(self):
        new_pr = np.asarray(self.pr).copy()
        for gt_label, pr_label in self.gt2pr.items():
            m = np.sum(self.pr == pr_label[0])
            new_pr[self.pr == pr_label[0]] = gt_label
        self.pr = np.asarray(new_pr).copy()

    def _set_boundaries(self):
        """Define boundaries for each segment from where sample."""
        cur_label = self.gt[0]
        mask = np.zeros(self.gt.shape, dtype=bool)
        for label_idx, label in enumerate(self.gt):
            if label == cur_label:
                mask[label_idx] = True
            else:
                self.bound_masks.append(mask)
                mask = np.zeros(self.gt.shape, dtype=bool)
                mask[label_idx] = True
                cur_label = label

    def _sampling(self):
        """Define mask for frames for which measure a score. And label if the segment defined correctly."""
        n_correct_segments = 0
        for mask in self.bound_masks:
            where = np.where(mask)[0]
            low = np.min(where)
            high = np.max(where)
            sampled_idxs = np.random.random_integers(low, high, self.sampling_ratio)
            n_corr_frames = np.sum(self.gt[sampled_idxs] == self.pr[sampled_idxs])
            n_correct_segments += n_corr_frames / self.sampling_ratio
            # if n_corr_frames > self.sampling_ratio / 2:
            #     n_correct_segments += 1

        precision = n_correct_segments / (self._K * self._n_videos)
        recall = n_correct_segments / len(self.bound_masks)
        f1 = 2 * (precision * recall) / (precision + recall + self._eps)
        self.f1_scores.append(f1)

        self._n_true_seg_all += n_correct_segments

        self._return['precision'] = [n_correct_segments, (self._K * self._n_videos)]
        self._return['recall'] = [n_correct_segments, len(self.bound_masks)]

    def f1(self):
        self._finish_init()
        for iteration in range(self.n_experiments):
            self._sampling()
        f1_mean = np.mean(self.f1_scores)
        logger.debug('f1 score: %f' % f1_mean)
        self._n_true_seg_all /= self.n_experiments
        self._return['precision'] = [self._n_true_seg_all, (self._K * self._n_videos)]
        self._return['recall'] = [self._n_true_seg_all, len(self.bound_masks)]
        self._return['mean_f1'] = [f1_mean, 1]


    def stat(self):
        return self._return



