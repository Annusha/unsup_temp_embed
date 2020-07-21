#!/usr/bin/env python

"""Implementation of dataset structure for dataloader from pytorch.
Two ways of training: with ground truth labels and with labels from current
segmentation given the algorithm"""


__author__ = 'Anna Kukleva'
__date__ = 'July 2020'

from torch.utils.data import Dataset
import numpy as np
import logging

class FeatureDataset(Dataset):
    def __init__(self, root_dir='', end='', subaction='coffee', videos=None,
                 features=None, regression=False, feature_list=None):
        """
        Filling out the _feature_list parameter. This is a list of [video name,
        index frame in video, ground truth, feature vector] for each frame of each video
        :param root_dir: root directory where exists folder 'ascii' with features or pure
         video files
        :param end: extension of files for current video representation (could be 'gz',
        'txt', 'avi')
        """
        self._logger = logging.getLogger('basic')
        self._logger.debug('FeatureDataset')

        self._root_dir = root_dir
        self._feature_list = None
        self._end = end
        self._old2new = {}
        self._videoname2idx = {}
        self._idx2videoname = {}
        self._videos = videos
        self._subaction = subaction
        self._features = features
        self._regression = regression

        self._feature_list = feature_list

        subactions = np.unique(self._feature_list[..., 2])
        for idx, old in enumerate(subactions):
            self._old2new[int(old)] = idx

    def index2name(self):
        return self._idx2videoname

    def __len__(self):
        return len(self._feature_list)

    def __getitem__(self, idx):
        name, frame_idx, gt_file, *features = self._feature_list[idx]
        gt_out = gt_file
        return np.asarray(features), gt_out

    def n_subact(self):
        return len(self._old2new)


