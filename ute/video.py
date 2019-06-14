#!/usr/bin/env python

"""Module with class for single video"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

from collections import Counter
import numpy as np
import math as m
import os
from os.path import join

from ute.utils.arg_pars import opt
from ute.utils.logging_setup import logger
from ute.utils.util_functions import dir_check
from ute.viterbi_utils.viterbi import Viterbi
from ute.viterbi_utils.grammar import Grammar

class Video(object):
    """Single video with respective for the algorithm parameters"""
    def __init__(self, path, K, *, gt=[], name='', start=0, with_bg=False):
        """
        Args:
            path (str): path to video representation
            K (int): number of subactivities in current video collection
            reset (bool): necessity of holding features in each instance
            gt (arr): ground truth labels
            gt_with_0 (arr): ground truth labels with SIL (0) label
            name (str): short name without any extension
            start (int): start index in mask for the whole video collection
        """
        self.iter = 0
        self.path = path
        self._K = K
        self.name = name

        self._likelihood_grid = None
        self._valid_likelihood = None
        self._theta_0 = 0.1
        self._subact_i_mask = np.eye(self._K)
        self.n_frames = 0
        self._features = None
        self.global_start = start
        self.global_range = None

        self.gt = gt
        self._gt_unique = np.unique(self.gt)

        self.features()
        self._check_gt()

        # counting of subactivities np.array
        self.a = np.zeros(self._K)
        # ordering, init with canonical ordering
        self._pi = list(range(self._K))
        self.inv_count_v = np.zeros(self._K - 1)
        # subactivity per frame
        self._z = []
        self._z_idx = []
        self._init_z_framewise()

        # temporal labels
        self.temp = None
        self._init_temporal_labels()

        # background
        self._with_bg = with_bg
        self.fg_mask = np.ones(self.n_frames, dtype=bool)
        if self._with_bg:
            self._init_fg_mask()

        self._subact_count_update()

        self.segmentation = {'gt': (self.gt, None)}

    def features(self):
        """Load features given path if haven't do it before"""
        if self._features is None:
            if opt.ext == 'npy':
                self._features = np.load(self.path)
            else:
                self._features = np.loadtxt(self.path)
            ######################################
            # fixed.order._coffee_mlp_!pose_full_vae0_time10.0_epochs60_embed20_n1_!ordering_gmm1_one_!gt_lr0.0001_lr_zeros_b0_v1_l0_c1_.pth.tar

            # self._features = self._features[1:, 1:]

            # if opt.data_type == 0 and opt.dataset == 'fs':
            #     self._features = self._features.T

            if opt.f_norm:  # normalize features
                mask = np.ones(self._features.shape[0], dtype=bool)
                for rdx, row in enumerate(self._features):
                    if np.sum(row) == 0:
                        mask[rdx] = False
                z = self._features[mask] - np.mean(self._features[mask], axis=0)
                z = z / np.std(self._features[mask], axis=0)
                self._features = np.zeros(self._features.shape)
                self._features[mask] = z
                self._features = np.nan_to_num(self.features())

            self.n_frames = self._features.shape[0]
            self._likelihood_grid = np.zeros((self.n_frames, self._K))
            self._valid_likelihood = np.zeros((self.n_frames, self._K), dtype=bool)
        return self._features

    def _check_gt(self):
        try:
            assert len(self.gt) == self.n_frames
        except AssertionError:
            print(self.path, '# gt and # frames does not match %d / %d' % (len(self.gt), self.n_frames))
            if abs(len(self.gt) - self.n_frames) > 50:
                if opt.data_type == 4:
                    os.remove(os.path.join(opt.gt, self.name))
                    os.remove(self.path)
                    try:
                        os.remove(os.path.join(opt.gt, 'mapping', 'gt%d%s.pkl' % (opt.frame_frequency, opt.gr_lev)))
                        os.remove(os.path.join(opt.gt, 'mapping', 'order%d%s.pkl' % (opt.frame_frequency, opt.gr_lev)))
                    except FileNotFoundError:
                        pass
                raise AssertionError
            else:
                min_n = min(len(self.gt), self.n_frames)
                self.gt = self.gt[:min_n]
                self.n_frames = min_n
                self._features = self._features[:min_n]
                self._likelihood_grid = np.zeros((self.n_frames, self._K))
                self._valid_likelihood = np.zeros((self.n_frames, self._K), dtype=bool)

    def _init_z_framewise(self):
        """Init subactivities uniformly among video frames"""
        # number of frames per activity
        step = m.ceil(self.n_frames / self._K)
        modulo = self.n_frames % self._K
        for action in range(self._K):
            # uniformly distribute remainder per actions if n_frames % K != 0
            self._z += [action] * (step - 1 * (modulo <= action) * (modulo != 0))
        self._z = np.asarray(self._z, dtype=int)
        try:
            assert len(self._z) == self.n_frames
        except AssertionError:
            logger.error('Wrong initialization for video %s', self.path)

    def _init_temporal_labels(self):
        self.temp = np.zeros(self.n_frames)
        for frame_idx in range(self.n_frames):
            self.temp[frame_idx] = frame_idx / self.n_frames
            # self.temp[frame_idx] = frame_idx

    def _init_fg_mask(self):
        indexes = [i for i in range(self.n_frames) if i % 2]
        self.fg_mask[indexes] = False
        # todo: have to check if it works correctly
        # since it just after initialization
        self._z[self.fg_mask == False] = -1

    def _subact_count_update(self):
        c = Counter(self._z)
        # logger.debug('%s: %s' % (self.name, str(c)))
        self.a = []
        for subaction in range(self._K):
            self.a += [c[subaction]]

    def update_indexes(self, total):
        self.global_range = np.zeros(total, dtype=bool)
        self.global_range[self.global_start: self.global_start + self.n_frames] = True

    def reset(self):
        """If features from here won't be in use anymore"""
        self._features = None

    def z(self, pi=None):
        """Construct z (framewise label assignments) from ordering and counting.
        Args:
            pi: order, if not given the current one is used
        Returns:
            constructed z out of indexes instead of actual subactivity labels
        """
        self._z = []
        self._z_idx = []
        if pi is None:
            pi = self._pi
        for idx, activity in enumerate(pi):
            self._z += [int(activity)] * self.a[int(activity)]
            self._z_idx += [idx] * self.a[int(activity)]
        if opt.bg:
            z = np.ones(self.n_frames, dtype=int) * -1
            z[self.fg_mask] = self._z
            self._z = z[:]
            z[self.fg_mask] = self._z_idx
            self._z_idx = z[:]
        assert len(self._z) == self.n_frames
        return np.asarray(self._z_idx)

    def update_z(self, z):
        self._z = np.asarray(z, dtype=int)
        self._subact_count_update()

    def likelihood_update(self, subact, scores, trh=None):
        # for all actions
        if subact == -1:
            self._likelihood_grid = scores
            if trh is not None:
                for trh_idx, single_trh in enumerate(trh):
                    self._valid_likelihood[:, trh_idx] = False
                    self._valid_likelihood[:, trh_idx] = scores[:, trh_idx] > single_trh
        else:
            # for all frames
            self._likelihood_grid[:, subact] = scores[:]
            if trh is not None:
                self._valid_likelihood[:, subact] = False
                self._valid_likelihood[:, subact] = scores > trh

    def valid_likelihood_update(self, trhs):
        for trh_idx, trh in enumerate(trhs):
            self._valid_likelihood[:, trh_idx] = False
            self._valid_likelihood[:, trh_idx] = self._likelihood_grid[:, trh_idx] > trh

    def save_likelihood(self):
        """Used for multiprocessing"""
        dir_check(os.path.join(opt.data, 'likelihood'))
        np.savetxt(os.path.join(opt.data, 'likelihood', self.name), self._likelihood_grid)
        # print(os.path.join(opt.data, 'likelihood', self.name))

    def load_likelihood(self):
        """Used for multiprocessing"""
        path_join = os.path.join(opt.data, 'likelihood', self.name)
        self._likelihood_grid = np.loadtxt(path_join)

    def get_likelihood(self):
        return self._likelihood_grid

    def _viterbi_inner(self, pi, save=False):
        grammar = Grammar(pi)
        if np.sum(self.fg_mask):
            viterbi = Viterbi(grammar=grammar, probs=(-1 * self._likelihood_grid[self.fg_mask]))
            viterbi.inference()
            viterbi.backward(strict=True)
            z = np.ones(self.n_frames, dtype=int) * -1
            z[self.fg_mask] = viterbi.alignment()
            score = viterbi.loglikelyhood()
        else:
            z = np.ones(self.n_frames, dtype=int) * -1
            score = -np.inf
        # viterbi.calc(z)
        if save:
            name = '%s_%s' % (str(pi), self.name)
            save_path = join(opt.output_dir, 'likelihood', name)
            with open(save_path, 'w') as f:
                # for pi_elem in pi:
                #     f.write('%d ' % pi_elem)
                # f.write('\n')
                f.write('%s\n' % str(score))
                for z_elem in z:
                    f.write('%d ' % z_elem)
        return z, score

    # @timing
    def viterbi(self, pi=None):

        if pi is None:
            pi = self._pi
        log_probs = self._likelihood_grid
        if np.max(log_probs) > 0:
            self._likelihood_grid = log_probs - (2 * np.max(log_probs))
        alignment, return_score = self._viterbi_inner(pi, save=True)
        self._z = np.asarray(alignment).copy()

        self._subact_count_update()

        name = str(self.name) + '_' + opt.log_str + 'iter%d' % self.iter + '.txt'
        np.savetxt(join(opt.output_dir, 'segmentation', name),
                   np.asarray(self._z), fmt='%d')
        # print('path alignment:', join(opt.data, 'segmentation', name))

        return return_score

    def update_fg_mask(self):
        self.fg_mask = np.sum(self._valid_likelihood, axis=1) > 0

    def resume(self):
        name = str(self.name) + '_' + opt.log_str + 'iter%d' % self.iter + '.txt'
        self._z = np.loadtxt(join(opt.output_dir, 'segmentation', name))
        self._subact_count_update()




