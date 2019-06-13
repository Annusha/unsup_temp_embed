#!/usr/bin/env python

"""Implementation of dataset structure for dataloader from pytorch.
Two ways of training: with ground truth labels and with labels from current
segmentation given the algorithm"""

__all__ = ['load_data']
__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

from torch.utils.data import Dataset
import torch
import numpy as np
import re
import logging

from ute.utils.mapping import GroundTruth
from ute.utils.arg_pars import opt
from ute.utils.logging_setup import logger
from ute.utils.util_functions import join_data


class FeatureDataset(Dataset):
    def __init__(self, root_dir, end, subaction='coffee', videos=None,
                 features=None):
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
        self.gt_map = GroundTruth()
        self._with_predictions()

        subactions = np.unique(self._feature_list[..., 2])
        for idx, old in enumerate(subactions):
            self._old2new[int(old)] = idx

    def index2name(self):
        return self._idx2videoname

    def _with_predictions(self):
        self._logger.debug('__init__')
        for video_idx, video in enumerate(self._videos):
            filename = re.match(r'[\.\/\w]*\/(\w+).\w+', video.path)
            if filename is None:
                logging.ERROR('Check paths videos, template to extract video name'
                              ' does not match')
            filename = filename.group(1)
            self._videoname2idx[filename] = video_idx
            self._idx2videoname[video_idx] = filename

            names = np.asarray([video_idx] * video.n_frames).reshape((-1, 1))
            idxs = np.asarray(list(range(0, video.n_frames))).reshape((-1, 1))
            if opt.gt_training:
                gt_file = np.asarray(video._gt).reshape((-1, 1))
            else:
                gt_file = np.asarray(video._z).reshape((-1, 1))
            if self._features is None:
                features = video.features()
            else:
                features = self._features[video.global_range]
            temp_feature_list = join_data(None,
                                          (names, idxs, gt_file, features),
                                          np.hstack)
            self._feature_list = join_data(self._feature_list,
                                           temp_feature_list,
                                           np.vstack)
        self._features = None

    def __len__(self):
        return len(self._feature_list)

    def __getitem__(self, idx):
        name, frame_idx, gt_file, *features = self._feature_list[idx]

        one_hot = np.zeros(self.n_subact())
        one_hot[self._old2new[int(gt_file)]] = 1
        gt_out = one_hot
        # features = torch.from_numpy(np.asarray(features))
        return np.asarray(features), gt_out, name

    def n_subact(self):
        return len(self._old2new)


def load_data(root_dir, end, subaction, videos=None, names=None, features=None):
    """Create dataloader within given conditions
    Args:
        root_dir: path to root directory with features
        end: extension of files
        subaction: complex activity
        videos: collection of object of class Video
        names: empty list as input to have opportunity to return dictionary with
            correspondences between names and indices
        features: features for the whole video collection
        regression: regression training
        vae: dataset for vae with incorporated relative time in features
    Returns:
        iterative dataloader
        number of subactions in current complex activity
    """
    logger.debug('create DataLoader')
    dataset = FeatureDataset(root_dir, end, subaction,
                             videos=videos,
                             features=features)
    if names is not None:
        names[0] = dataset.index2name()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=(not opt.save_embed_feat),
                                             num_workers=opt.num_workers)

    return dataloader, dataset.n_subact()


