#!/usr/bin/env python

""" Some processing of ground truth information.

gt: Load ground truth labels for each video and save in gt dict.
label2index, index2label: As well unique mapping between name of complex
    actions and their order index.
define_K: define number of subactions from ground truth labeling
"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import os
import numpy as np
import pickle

from ute.utils.arg_pars import opt
from ute.utils.util_functions import timing, dir_check
from ute.utils.logging_setup import logger


class GroundTruth:
    def __init__(self, frequency=1):
        self.label2index = {}
        self.index2label = {}
        self.frequency = frequency

        self.gt = {}
        self.order = {}

    def create_mapping(self):
        root = opt.mapping_dir
        filename = 'mapping%s.txt' % opt.gr_lev

        with open(os.path.join(root, filename), 'r') as f:
            for line in f:
                idx, class_name = line.split()
                idx = int(idx)
                self.label2index[class_name] = idx
                self.index2label[idx] = class_name
            if not opt.bg and -1 in self.label2index:
                # change bg label from -1 to positive number
                new_bg_idx = max(self.index2label) + 1
                del self.index2label[self.label2index[-1]]
                self.label2index[-1] = new_bg_idx
                self.index2label[new_bg_idx] = -1

    @staticmethod
    def load_obj(name):
        path = os.path.join(opt.mapping_dir, '%s.pkl' % name)
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            return None

    @staticmethod
    def save_obj(obj, name):
        dir_check(opt.mapping_dir)
        path = os.path.join(opt.mapping_dir, '%s.pkl' % name)
        with open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @timing
    def load_gt(self):
        self.gt = self.load_obj('gt%d%s' % (self.frequency, opt.gr_lev))
        self.order = self.load_obj('order%d%s' % (self.frequency, opt.gr_lev))

        if self.gt is None or self.order is None:
            logger.debug('cannot load -> create mapping')
            self.gt, self.order = {}, {}
            for root, dirs, files in os.walk(opt.gt):
                for filename in files:
                    with open(os.path.join(root, filename), 'r') as f:
                        labels = []
                        local_order = []
                        curr_lab = -1
                        start, end = 0, 0
                        for line_idx, line in enumerate(f):
                            if line_idx % self.frequency:
                                continue
                            line = line.split()[0]
                            try:
                                labels.append(self.label2index[line])
                                if curr_lab != labels[-1]:
                                    if curr_lab != -1:
                                        local_order.append([curr_lab, start, end])
                                    curr_lab = labels[-1]
                                    start = end
                                end += 1
                            except KeyError:
                                break
                        else:
                            # executes every times when "for" wasn't interrupted by break
                            self.gt[filename] = np.array(labels)
                            # add last labels

                            local_order.append([curr_lab, start, end])
                            self.order[filename] = local_order
            self.save_obj(self.gt, 'gt%d%s' % (self.frequency, opt.gr_lev))
            self.save_obj(self.order, 'order%d%s' % (self.frequency, opt.gr_lev))
        else:
            logger.debug('successfully loaded')

    def define_K(self, subaction):
        """Define number of subactions from ground truth labeling

        Args:
            subaction (str): name of complex activity
        Returns:
            number of subactions
        """
        uniq_labels = set()
        for filename, labels in self.gt.items():
            if subaction in filename:
                uniq_labels = uniq_labels.union(labels)
        if -1 in uniq_labels:
            return len(uniq_labels) - 1
        else:
            return len(uniq_labels)

    def sparse_gt(self):
        for key, val in self.gt.items():
            sparse_segm = [i for i in val[::10]]
            self.gt[key] = sparse_segm

    def load_mapping(self):
        logger.debug('load or create mapping')
        self.create_mapping()
        self.load_gt()




