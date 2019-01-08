#!/usr/bin/env python

"""Class for visualization of trained embedding: PCA or t-SNE methods are used
for dimensionality reduction."""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.util_functions import join_data
from os.path import join
import matplotlib.style
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable

from utils.arg_pars import opt
from utils.util_functions import dir_check, timing


class Visual(object):
    def __init__(self, mode='pca', dim=2, full=True, save=False):

        self._mode = mode
        self._model = None
        self._dim = dim
        self._data = None
        self._labels = None
        self._sizes = []
        self._counter = 0
        self._result = None
        self.size = 1  # size of dots
        self._full = full
        self._save = save

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = join_data(self._data, new_data, np.vstack)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, new_labels):
        self._labels = join_data(self._labels, new_labels, np.hstack)
        self._sizes += [self.size] * len(new_labels)

    @timing
    def fit_data(self, reduce=None):
        if self._mode == 'pca':
            self._model = PCA(n_components=self._dim, random_state=opt.seed)
        if self._mode == 'tsne':
            self._model = TSNE(n_components=self._dim, perplexity=15, random_state=opt.seed)
        if self._full:
            self._result = self._model.fit_transform(self._data)
        else:
            self._model.fit(self._data[:reduce])
            self._result = self._model.transform(self._data)

    def plot(self, iter=0, show=True, gt_plot=0, prefix=''):
        if iter is not None:
            self._counter = iter
        plt.scatter(self._result[..., 0], self._result[..., 1],
                    c=self._labels, s=self._sizes, alpha=0.5)
        plt.grid(True)
        if self._save:
            # plt.figure(figsize=(1))
            dir_check(join(opt.dataset_root, 'plots'))
            dir_check(join(opt.dataset_root, 'plots', opt.subaction))
            # name = ['iter%d_' % self._counter, 'gt_'][gt_plot]
            name = prefix + '%s_%s_' % (opt.subaction,  opt.model_name)
            name += '_%s.png' % self._mode
            folder_name = opt.log_str
            dir_check(join(opt.dataset_root, 'plots', opt.subaction, folder_name))
            plt.savefig(join(opt.dataset_root, 'plots', opt.subaction,
                             folder_name, name), dpi=400)
            # else:
            #     plt.savefig(join(opt.dataset_root, 'plots', opt.subaction, name), dpi=400)
        if show:
            plt.show()

    def reset(self):
        plt.clf()
        self._counter += 1
        self._data = None
        self._labels = None
        self._sizes = []
        self.size = 1

    def fit(self, data, labels, prefix):
        self._data = data
        self._labels = labels
        self._sizes += [self.size] * len(labels)
        self.fit_data()
        self.plot(show=False, prefix=prefix)
        self.reset()

def bounds(segm):
    start_label = segm[0]
    start_idx = 0
    idx = 0
    while idx < len(segm):
        try:
            while start_label == segm[idx]:
                idx += 1
        except IndexError:
            yield start_idx, idx, start_label
            break

        yield start_idx, idx, start_label
        start_idx = idx
        start_label = segm[start_idx]


def plot_segm(path, segmentation, colors, name=''):
    mpl.style.use('classic')
    fig = plt.figure(figsize=(16, 4))
    plt.axis('off')
    plt.title(name, fontsize=20)
    # plt.subplots_adjust(top=0.9, hspace=0.6)
    gt_segm, _ = segmentation['gt']
    ax_idx = 1
    plots_number = len(segmentation)
    ax = fig.add_subplot(plots_number, 1, ax_idx)
    ax.set_ylabel('GT', fontsize=30, rotation=0, labelpad=40, verticalalignment='center')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # make_axes_area_auto_adjustable(ax)
    # plt.title('gt', fontsize=20)
    v_len = len(gt_segm)
    for start, end, label in bounds(gt_segm):
        ax.axvspan(start / v_len, end / v_len, facecolor=colors[label], alpha=1.0)
    for key, (segm, label2gt) in segmentation.items():
        if key in ['gt', 'cl']:
            continue
        ax_idx += 1
        ax = fig.add_subplot(plots_number, 1, ax_idx)
        ax.set_ylabel('OUTPUT', fontsize=30, rotation=0, labelpad=60, verticalalignment='center')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # make_axes_area_auto_adjustable(ax)
        segm = list(map(lambda x: label2gt[x], segm))
        for start, end, label in bounds(segm):
            ax.axvspan(start / v_len, end / v_len, facecolor=colors[label], alpha=1.0)


    fig.savefig(path, transparent=False)
