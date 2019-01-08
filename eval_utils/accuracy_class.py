#!/usr/bin/env python

"""Module for computing evaluation metrics"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import numpy as np
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment

from utils.arg_pars import opt
from utils.logging_setup import logger


class Accuracy(object):
    """ Implementation of evaluation metrics for unsupervised learning.

    Since it's unsupervised learning relations between ground truth labels
    and output segmentation should be found.
    Hence the Hungarian method was used and labeling which gives us
    the best score is used as a result.
    """
    def __init__(self, n_frames=1):
        """
        Args:
            n_frames: frequency of sampling,
                in case of it's equal to 1 => dense sampling
        """
        self._n_frames = n_frames
        self._reset()

        self._predicted_labels = None
        self._gt_labels_subset = None
        self._gt_labels = None
        self._boundaries = None
        # all frames used for alg without any subsampling technique
        self._indices = None

        self._frames_overall = 0
        self._frames_true_pr = 0
        self._average_score = 0
        self._processed_number = 0
        self._classes_MoF = {}
        self._classes_IoU = {}
        # keys - gt, values - pr
        self.exclude = {}

        self._logger = logger
        self._return = {}

    def _reset(self):
        self._n_clusters = 0

        self._gt_label2index = {}
        self._gt_index2label = {}
        self._pr_label2index = {}
        self._pr_index2label = {}

        self._voting_table = []
        self._gt2cluster = defaultdict(list)
        self._acc_per_gt_class = {}

        self.exclude = {}

    @property
    def predicted_labels(self):
        return self._predicted_labels

    @predicted_labels.setter
    def predicted_labels(self, labels):
        self._predicted_labels = np.array(labels)
        self._reset()

    @property
    def gt_labels(self):
        return self._gt_labels_subset

    @gt_labels.setter
    def gt_labels(self, labels):
        self._gt_labels = np.array(labels)
        self._gt_labels_subset = self._gt_labels[:]
        self._indices = list(range(len(self._gt_labels)))

    @property
    def params(self):
        """
        boundaries: if frames samples from segments we need to know boundaries
            of these segments to fulfill them after
        indices: frames extracted for whatever and indeed evaluation
        """
        return self._boundaries, self._indices

    @params.setter
    def params(self, params):
        self._boundaries = params[0]
        self._indices = params[1]
        self._gt_labels_subset = self._gt_labels[self._indices]

    def _create_voting_table(self):
        """Filling table with assignment scores.

        Create table which represents paired label assignments, i.e. each
        cell comprises score for corresponding label assignment"""
        size = max(len(np.unique(self._gt_labels_subset)),
                   len(np.unique(self._predicted_labels)))
        self._voting_table = np.zeros((size, size))

        for idx_gt, gt_label in enumerate(np.unique(self._gt_labels_subset)):
            self._gt_label2index[gt_label] = idx_gt
            self._gt_index2label[idx_gt] = gt_label

        if len(self._gt_label2index) < size:
            for idx_gt in range(len(np.unique(self._gt_labels_subset)), size):
                gt_label = idx_gt
                while gt_label in self._gt_label2index:
                    gt_label += 1
                self._gt_label2index[gt_label] = idx_gt
                self._gt_index2label[idx_gt] = gt_label

        for idx_pr, pr_label in enumerate(np.unique(self._predicted_labels)):
            self._pr_label2index[pr_label] = idx_pr
            self._pr_index2label[idx_pr] = pr_label

        if len(self._pr_label2index) < size:
            for idx_pr in range(len(np.unique(self._predicted_labels)), size):
                pr_label = idx_pr
                while pr_label in self._pr_label2index:
                    pr_label += 1
                self._pr_label2index[pr_label] = idx_pr
                self._pr_index2label[idx_pr] = pr_label

        for idx_gt, gt_label in enumerate(np.unique(self._gt_labels_subset)):
            if gt_label in list(self.exclude.keys()):
                continue
            gt_mask = self._gt_labels_subset == gt_label
            for idx_pr, pr_label in enumerate(np.unique(self._predicted_labels)):
                if pr_label in list(self.exclude.values()):
                    continue
                self._voting_table[idx_gt, idx_pr] = \
                    np.sum(self._predicted_labels[gt_mask] == pr_label, dtype=float)
        for key, val in self.exclude.items():
            # works only if one pair in exclude
            assert len(self.exclude) == 1
            try:
                self._voting_table[self._gt_label2index[key], self._pr_label2index[val[0]]] = size * np.max(self._voting_table)
            except KeyError:
                logger.debug('No background!')
                self._voting_table[self._gt_label2index[key], -1] = size * np.max(self._voting_table)
                self._pr_index2label[size - 1] = val[0]
                self._pr_label2index[val[0]] = size - 1

    def _create_correspondences(self, method='hungarian', optimization='max'):
        """ Find output labels which correspond to ground truth labels.

        Hungarian method finds one-to-one mapping: if there is squared matrix
        given, then for each output label -> gt label. If not, some labels will
        be without correspondences.
        Args:
            method: hungarian or max
            optimization: for hungarian method usually min problem but here
                is max, hence convert to min
            where: if some actions are not in the video collection anymore
        """
        if method == 'hungarian':
            try:
                assert self._voting_table.shape[0] == self._voting_table.shape[1]
            except AssertionError:
                self._logger.debug('voting table non squared')
                raise AssertionError('bum tss')
            if optimization == 'max':
                # convert max problem to minimization problem
                self._voting_table *= -1
            x, y = linear_sum_assignment(self._voting_table)
            for idx_gt, idx_pr in zip(x, y):
                self._gt2cluster[self._gt_index2label[idx_gt]] = [self._pr_index2label[idx_pr]]
        if method == 'max':
            # maximum voting, won't create exactly one-to-one mapping
            max_responses = np.argmax(self._voting_table, axis=0)
            for idx, c in enumerate(max_responses):
                # c is index of gt label
                # idx is predicted cluster label
                self._gt2cluster[self._gt_index2label[c]].append(idx)


    def _fulfill_segments(self):
        """If was used frame sampling then anyway we need to get assignment
        for each frame"""
        self._full_predicted_labels = []
        for idx, slice in enumerate(range(0, len(self._predicted_labels), self._n_frames)):
            start, end = self._boundaries[idx]
            label_counter = Counter(self._predicted_labels[slice: slice + self._n_frames])
            win_label = label_counter.most_common(1)[0][0]
            self._full_predicted_labels += [win_label] * (end - start + 1)
        self._full_predicted_labels = np.asarray(self._full_predicted_labels)

    def mof(self, with_segments=False, old_gt2label=None, optimization='max'):
        """ Compute mean over frames (MoF) for current labeling.

        Args:
            with_segments: if frame sampling was used
            old_gt2label: MoF for given gt <-> output labels correspondences
            optimization: inside hungarian method
            where: see _create_correspondences method

        Returns:

        """
        self._n_clusters = len(np.unique(self._predicted_labels))
        self._create_voting_table()
        self._create_correspondences(optimization=optimization)
        self._logger.debug('# gt_labels: %d   # pr_labels: %d' %
                           (len(np.unique(self._gt_labels_subset)),
                            len(np.unique(self._predicted_labels))))
        self._logger.debug('Correspondences: segmentation to gt : '
                           + str([('%d: %d' % (value[0], key)) for (key, value) in
                                  sorted(self._gt2cluster.items(), key=lambda x: x[-1])]))
        if with_segments:
            self._fulfill_segments()
        else:
            self._full_predicted_labels = self._predicted_labels

        old_frames_true = 0
        self._classes_MoF = {}
        self._classes_IoU = {}
        excluded_total = 0
        for gt_label in np.unique(self._gt_labels):
            true_defined_frame_n = 0.
            union = 0
            gt_mask = self._gt_labels == gt_label
            # no need the loop since only one label should be here
            # i.e. one-to-one mapping, but i'm lazy
            for cluster in self._gt2cluster[gt_label]:
                true_defined_frame_n += np.sum(self._full_predicted_labels[gt_mask] == cluster,
                                               dtype=float)
                pr_mask = self._full_predicted_labels == cluster
                union += np.sum(gt_mask | pr_mask)
            if old_gt2label is not None:
                old_true_defined_frame_n = 0.
                for cluster in old_gt2label[gt_label]:
                    old_true_defined_frame_n += np.sum(self._full_predicted_labels[gt_mask] == cluster,
                                                       dtype=float)
                old_frames_true += old_true_defined_frame_n

            self._classes_MoF[gt_label] = [true_defined_frame_n, np.sum(gt_mask)]
            self._classes_IoU[gt_label] = [true_defined_frame_n, union]

            if gt_label in self.exclude:
                if opt.zeros == False and gt_label == 0:
                    continue
                excluded_total += np.sum(gt_mask)
            else:
                self._frames_true_pr += true_defined_frame_n

        self._frames_overall = len(self._gt_labels)
        self._frames_overall -= excluded_total
        return old_frames_true, self._frames_overall

    def mof_classes(self):
        average_class_mof = 0
        total_true = 0
        total = 0
        for key, val in self._classes_MoF.items():
            true_frames, all_frames = val
            logger.debug('label %d: %f  %d / %d' % (key, true_frames / all_frames,
                                                    true_frames, all_frames))
            average_class_mof += true_frames / all_frames
            total_true += true_frames
            total += all_frames
        average_class_mof /= len(self._classes_MoF)
        logger.debug('average class mof: %f' % average_class_mof)
        self._return['mof'] = [self._frames_true_pr, self._frames_overall]
        self._return['mof_bg'] = [total_true, total]
        if opt.bg:
            logger.debug('mof with bg: %f' % (total_true / total))

    def iou_classes(self):
        average_class_iou = 0
        excluded_iou = 0
        for key, val in self._classes_IoU.items():
            true_frames, union = val
            logger.debug('label %d: %f  %d / %d' % (key, true_frames / union,
                                                    true_frames, union))
            if key not in self.exclude:
                average_class_iou += true_frames / union
            else:
                excluded_iou += true_frames / union
        average_iou_without_exc = average_class_iou / \
                                  (len(self._classes_IoU) - len(self.exclude))
        average_iou_with_exc = (average_class_iou + excluded_iou) / \
                               len(self._classes_IoU)
        logger.debug('average IoU: %f' % average_iou_without_exc)
        self._return['iou'] = [average_class_iou,
                               len(self._classes_IoU) - len(self.exclude)]
        self._return['iou_bg'] = [average_class_iou + excluded_iou,
                                  len(self._classes_IoU) - len(self.exclude)]
        if self.exclude:
            logger.debug('average IoU with bg: %f' % average_iou_with_exc)


    def mof_val(self):
        self._logger.debug('frames true: %d\tframes overall : %d' %
                           (self._frames_true_pr, self._frames_overall))
        return float(self._frames_true_pr) / self._frames_overall

    def frames(self):
        return self._frames_true_pr

    def stat(self):
        return self._return
