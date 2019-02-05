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
import copy
import pickle

from utils.arg_pars import opt
from utils.util_functions import timing, dir_check


class GroundTruth:
    def __init__(self):
        self.label2index = {}
        self.index2label = {}

        self.gt = {}
        self.order = {}

    def create_mapping(self):
        root = os.path.join(opt.gt, 'mapping')
        if opt.high:
            filename = 'map_high.txt'
        else:
            filename = 'mapping.txt'

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
        path = os.path.join(opt.gt, 'mapping', '%s.pkl' % name)
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            return None

    @staticmethod
    def save_obj(obj, name):
        dir_check(os.path.join(opt.gt, 'mapping'))
        path = os.path.join(opt.gt, 'mapping', '%s.pkl' % name)
        with open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @timing
    def load_gt(self):
        if opt.high:
            self.gt = self.load_obj('gt_high')
            self.order = self.load_obj('order_high')
        else:
            self.gt = self.load_obj('gt')
            self.order = self.load_obj('order')

        if self.gt is None or self.order is None:
            self.gt, self.order = {}, {}
            for filename in os.listdir(opt.gt):
                if os.path.isdir(os.path.join(opt.gt, filename)):
                    continue
                with open(os.path.join(opt.gt, filename), 'r') as f:
                    labels = []
                    local_order = []
                    curr_lab = -1
                    start, end = 0, 0
                    for line in f:
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
            if opt.high:
                self.save_obj(self.gt, 'gt_high')
                self.save_obj(self.order, 'order_high')
            else:
                self.save_obj(self.gt, 'gt')
                self.save_obj(self.order, 'order')

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
        self.create_mapping()
        self.load_gt()




