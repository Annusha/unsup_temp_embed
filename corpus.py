#!/usr/bin/env python

"""Module with Corpus class. There are methods for each step of the alg for the
whole video collection of one complex activity. See pipeline."""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import numpy as np
import os
import os.path as ops
import torch
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.mixture import GaussianMixture
import time
from sklearn.cluster import MiniBatchKMeans

from video import Video
from probabilistic_utils.mallow import Mallow
from probabilistic_utils.slice_sampling import slice_sampling
from models import mlp, tcn
from utils.arg_pars import opt
from utils.logging_setup import logger
from eval_utils.accuracy_class import Accuracy
from utils.mapping import GroundTruth
from utils.util_functions import join_data, timing, dir_check
from utils.visualization import Visual, plot_segm
from probabilistic_utils.gmm_utils import AuxiliaryGMM, GMM_trh
from eval_utils.f1_score import F1Score
from models.dataset_loader import load_reltime
from models.training_embed import load_model, training


class Corpus(object):
    def __init__(self, Q, subaction='coffee'):
        """
        Args:
            Q: number of Gaussian components in each mixture
            subaction: current name of complex activity
        """
        np.random.seed(opt.seed)
        self.gt_map = GroundTruth(frequency=opt.frame_frequency)
        self.gt_map.load_mapping()
        self._K = self.gt_map.define_K(subaction=subaction)
        logger.debug('%s  subactions: %d' % (subaction, self._K))
        self.iter = -1
        self.return_stat = {}

        self._acc_old = 0
        self._videos = []
        self._subaction = subaction
        # init with ones for consistency with first measurement of MoF
        self._subact_counter = np.ones(self._K)
        # number of gaussian in each mixture
        self._Q = 1 if opt.gt_training else Q
        self._gaussians = {}
        self._mallow = Mallow(self._K)
        self._inv_count_stat = np.zeros(self._K)
        self._embedding = None
        self._gt2label = None
        self._label2gt = {}

        self._with_bg = opt.bg
        self._total_fg_mask = None

        # multiprocessing for sampling activities for each video
        self._features = None
        self._embedded_feat = None
        self._init_videos()
        logger.debug('min: %f  max: %f  avg: %f' %
                     (np.min(self._features),
                      np.max(self._features),
                      np.mean(self._features)))

        if opt.ordering:
            self.rho_sampling()

        # to save segmentation of the videos
        dir_check(os.path.join(opt.data, 'segmentation'))
        dir_check(os.path.join(opt.data, 'likelihood'))
        self.vis = None  # visualization tool

    def _init_videos(self):
        logger.debug('.')
        gt_stat = Counter()
        for root, dirs, files in os.walk(os.path.join(opt.data, opt.subfolder)):
            if files:
                for filename in files:
                    # pick only videos with certain complex action
                    # (ex: just concerning coffee)
                    if self._subaction in filename:
                        if opt.test_set:
                            if opt.reduced:
                                opt.reduced = opt.reduced - 1
                                continue
                        if opt.dataset == 'fs':
                            gt_name = filename[:-(len(opt.ext) + 1)] + '.txt'
                        else:
                            match = re.match(r'(\w*)\.\w*', filename)
                            gt_name = match.group(1)
                        # use extracted features from pretrained on gt embedding
                        if opt.load_embed_feat:
                            path = os.path.join(opt.data, 'embed', opt.subaction,
                                                opt.resume_str % opt.subaction) + '_%s' % gt_name
                        else:
                            path = os.path.join(root, filename)
                        start = 0 if self._features is None else self._features.shape[0]
                        try:
                            video = Video(path, K=self._K,
                                          gt=self.gt_map.gt[gt_name],
                                          name=gt_name,
                                          start=start,
                                          with_bg=self._with_bg)
                        except AssertionError:
                            logger.debug('Assertion Error: %s' % gt_name)
                            continue
                        self._features = join_data(self._features, video.features(),
                                                   np.vstack)

                        video.reset()  # to not store second time loaded features
                        self._videos.append(video)
                        # accumulate statistic for inverse counts vector for each video
                        gt_stat.update(self.gt_map.gt[gt_name])
                        if opt.reduced:
                            if len(self._videos) > opt.reduced:
                                break

                        if opt.feature_dim > 100:
                            if len(self._videos) % 20 == 0:
                                logger.debug('loaded %d videos' % len(self._videos))

        # update global range within the current collection for each video
        for video in self._videos:
            video.update_indexes(len(self._features))
        logger.debug('gt statistic: %d videos ' % len(self._videos) + str(gt_stat))
        self._update_fg_mask()

    def _update_fg_mask(self):
        logger.debug('.')
        if self._with_bg:
            self._total_fg_mask = np.zeros(len(self._features), dtype=bool)
            for video in self._videos:
                self._total_fg_mask[np.nonzero(video.global_range)[0][video.fg_mask]] = True
        else:
            self._total_fg_mask = np.ones(len(self._features), dtype=bool)

    def regression_training(self):
        if opt.load_embed_feat:
            logger.debug('load precomputed features')
            self._embedded_feat = self._features
            return

        logger.debug('.')

        dataloader = load_reltime(videos=self._videos,
                                  features=self._features)

        if opt.model_name == 'mlp':
            model, loss, optimizer = mlp.create_model()
        if opt.model_name == 'tcn':
            model, loss, optimizer = tcn.create_model()
        if opt.resume:
            model.load_state_dict(load_model(name=opt.model_name))
            self._embedding = model
        else:
            self._embedding = training(dataloader, opt.epochs,
                                       save=opt.save_model,
                                       model=model,
                                       loss=loss,
                                       optimizer=optimizer,
                                       name=opt.model_name)

        self._embedding = self._embedding.cpu()

        unshuffled_dataloader = load_reltime(videos=self._videos,
                                             features=self._features,
                                             shuffle=False)

        gt_relative_time = None
        relative_time = None
        if opt.model_name == 'mlp':
            for batch_features, batch_gtreltime in unshuffled_dataloader:
                if self._embedded_feat is None:
                    self._embedded_feat = batch_features
                else:
                    self._embedded_feat = torch.cat((self._embedded_feat, batch_features), 0)

                batch_gtreltime = batch_gtreltime.numpy().reshape((-1, 1))
                gt_relative_time = join_data(gt_relative_time, batch_gtreltime, np.vstack)

            relative_time = self._embedding(self._embedded_feat.float()).detach().numpy().reshape((-1, 1))

            self._embedded_feat = self._embedding.embedded(self._embedded_feat.float()).detach().numpy()
            self._embedded_feat = np.squeeze(self._embedded_feat)

        if opt.model_name == 'tcn':
            for batch_features, batch_gtreltime in unshuffled_dataloader:
                batch_gtreltime = batch_gtreltime.numpy()[:, -1].reshape((-1, 1))
                gt_relative_time = join_data(gt_relative_time, batch_gtreltime, np.vstack)

                batch_rel_time = self._embedding(batch_features.float()).detach().numpy()
                batch_rel_time = batch_rel_time[:, -1].reshape((-1, 1))
                relative_time = join_data(relative_time, batch_rel_time, np.vstack)

                embedded_feat = self._embedding.embedded(batch_features.float()).detach().numpy()
                self._embedded_feat = join_data(self._embedded_feat, embedded_feat, np.vstack)

        if opt.save_embed_feat:
            self.save_embed_feat()

        mse = np.sum((gt_relative_time - relative_time)**2)
        mse = mse / len(relative_time)
        logger.debug('MLP training: MSE: %f' % mse)
        hist, bin_edges = np.histogram(relative_time, bins=np.linspace(0, 1, 101))
        logger.debug('Histogram: %s' % str(hist))
        logger.debug('%s' % str(bin_edges))
        self.hist = hist

    @timing
    def gaussian_clustering(self):
        logger.debug('.')
        gmm = GaussianMixture(n_components=self._K,
                              covariance_type='full',
                              random_state=opt.seed,
                              reg_covar=opt.reg_cov)

        ########################################################################
        #####  concat with respective relative time label
        if opt.rt_cl_concat:
            temp_embedd = self._embedded_feat.copy()
            long_rel_time = []
            for video in self._videos:
                long_rel_time += list(video.temp)
            long_rel_time = np.asarray(long_rel_time).reshape((-1, 1))
            self._embedded_feat = join_data(self._embedded_feat,
                                            long_rel_time,
                                            np.hstack)
            assert self._embedded_feat.shape[1] == (opt.embed_dim + 1)
        ########################################################################

        gmm.fit(self._embedded_feat)
        predicted_labels = gmm.predict(self._embedded_feat)
        post_probs = np.log(gmm.predict_proba(self._embedded_feat))
        post_probs = np.nan_to_num(post_probs)

        if opt.rt_cl_concat:
            self._embedded_feat = temp_embedd

        accuracy = Accuracy()
        long_gt = []
        long_temp = []
        for video in self._videos:
            long_gt += list(video.gt)
            long_temp += list(video.temp)
        long_temp = np.array(long_temp)

        predicted_labels_copy = np.asarray(predicted_labels).copy()
        time2label = {}
        for label in np.unique(predicted_labels_copy):
            cluster_mask = predicted_labels_copy == label
            r_time = np.mean(long_temp[self._total_fg_mask][cluster_mask])
            time2label[r_time] = label

        ordered_labels = []
        for time_idx, sorted_time in enumerate(sorted(time2label)):
            label = time2label[sorted_time]
            ordered_labels.append(label)
            predicted_labels_copy[predicted_labels == label] = time_idx

        labels_with_bg = np.ones(len(self._total_fg_mask)) * -1
        labels_with_bg[self._total_fg_mask] = predicted_labels_copy

        logger.debug('Order of labels: %s, %s' % (str(ordered_labels), str(sorted(time2label))))
        accuracy.predicted_labels = labels_with_bg
        accuracy.gt_labels = long_gt
        old_mof, total_fr = accuracy.mof()
        self._gt2label = accuracy._gt2cluster
        for key, val in self._gt2label.items():
            try:
                self._label2gt[val[0]] = key
            except IndexError:
                pass

        logger.debug('MoF val: ' + str(accuracy.mof_val()))
        logger.debug('old MoF val: ' + str(float(old_mof) / total_fr))

        ########################################################################
        # VISUALISATION
        # if opt.vis:
        #     vis = Visual(mode=opt.vis_mode, save=True)
        #     vis.fit(self._embedded_feat, labels_with_bg, 'gmm_')
        ########################################################################


        logger.debug('Update video z and likelihood')
        post_probs = post_probs[:, ordered_labels]  # order with respect to label ordering
        labels_with_bg[labels_with_bg == self._K] = -1
        for video in self._videos:
            video.update_z(labels_with_bg[video.global_range])
            video.likelihood_update(subact=-1,
                                    scores=post_probs[video.global_range])

            video.segmentation['cl'] = (video._z, self._label2gt)


    @timing
    def _gaussians_fit(self):
        """ Fit GMM to video features.

        Define subset of video collection and fit on it gaussian mixture model.
        If idx_exclude = -1, then whole video collection is used for comprising
        the subset, otherwise video collection excluded video with this index.
        Args:
            idx_exclude: video to exclude (-1 or int in range(0, #_of_videos))
            save: in case of mp lib all computed likelihoods are saved on disk
                before the next step
        """
        # logger.debug('Excluded: %d' % idx_exclude)
        for k in range(self._K):
            gmm = GaussianMixture(n_components=self._Q,
                                  covariance_type='full',
                                  max_iter=150,
                                  random_state=opt.seed,
                                  reg_covar=opt.reg_cov)
            total_indexes = np.zeros(len(self._features), dtype=np.bool)
            for idx, video in enumerate(self._videos):
                if opt.gt_training:
                    indexes = np.where(np.asarray(video._gt) == self._label2gt[k])[0]
                else:
                    indexes = np.where(np.asarray(video._z) == k)[0]
                if len(indexes) == 0:
                    continue
                temp = np.zeros(video.n_frames, dtype=np.bool)
                temp[indexes] = True
                total_indexes[video.global_range] = temp

            total_indexes = np.array(total_indexes, dtype=np.bool)
            if opt.load_embed_feat:
                feature = self._features[total_indexes, :]
            else:
                feature = self._embedded_feat[total_indexes, :]
            time1 = time.time()
            try:
                gmm.fit(feature)
            except ValueError:
                gmm = AuxiliaryGMM()
            time2 = time.time()
            # logger.debug('fit gmm %0.6f %d ' % ((time2 - time1), len(feature))
            #              + str(gmm.converged_))

            self._gaussians[k] = gmm

        if opt.bg:
            # with bg model I assume that I use only one component
            assert self._Q == 1
            for gm_idx, gmm in self._gaussians.items():
                self._gaussians[gm_idx] = GMM_trh(gmm)

    @timing
    def gaussian_model(self):
        logger.debug('Fit Gaussian Mixture Model to the whole dataset at once')
        self._gaussians_fit()
        for video_idx in range(len(self._videos)):
            self._video_likelihood_grid(video_idx)

        if opt.bg:
            scores = None
            for video in self._videos:
                scores = join_data(scores, video.get_likelihood(), np.vstack)
            # ------ max ------
            # bg_trh_score = np.max(scores, axis=0)
            # logger.debug('bg_trh_score: %s' % str(bg_trh_score))
            # ------ max ------

            bg_trh_score = np.sort(scores, axis=0)[int((opt.bg_trh / 100) * scores.shape[0])]

            bg_trh_set = []
            for action_idx in range(self._K):
                new_bg_trh = self._gaussians[action_idx].mean_score - bg_trh_score[action_idx]
                self._gaussians[action_idx].update_trh(new_bg_trh=new_bg_trh)
                bg_trh_set.append(new_bg_trh)

            logger.debug('new bg_trh: %s' % str(bg_trh_set))
            trh_set = []
            for action_idx in range(self._K):
                trh_set.append(self._gaussians[action_idx].trh)
            for video in self._videos:
                video.valid_likelihood_update(trh_set)

    def _video_likelihood_grid(self, video_idx):
        video = self._videos[video_idx]
        if opt.load_embed_feat:
            features = self._features[video.global_range]
        else:
            features = self._embedded_feat[video.global_range]
        for subact in range(self._K):
            scores = self._gaussians[subact].score_samples(features)
            if opt.bg:
                video.likelihood_update(subact, scores,
                                        trh=self._gaussians[subact].trh)
            else:
                video.likelihood_update(subact, scores)
        if opt.save_likelihood:
            video.save_likelihood()

    def clustering(self):
        logger.debug('.')
        np.random.seed(opt.seed)

        kmean = MiniBatchKMeans(n_clusters=self._K,
                                 random_state=opt.seed,
                                 batch_size=50)

        ########################################################################
        #####  concat with respective relative time label
        if opt.rt_cl_concat:
            temp_embedd = self._embedded_feat.copy()
            long_rel_time = []
            for video in self._videos:
                long_rel_time += list(video.temp)
            long_rel_time = np.asarray(long_rel_time).reshape((-1, 1))
            self._embedded_feat = join_data(self._embedded_feat,
                                            long_rel_time,
                                            np.hstack)
            assert self._embedded_feat.shape[1] == (opt.embed_dim + 1)
        ########################################################################

        kmean.fit(self._embedded_feat[self._total_fg_mask])

        if opt.rt_cl_concat:
            self._embedded_feat = temp_embedd

        accuracy = Accuracy()
        long_gt = []
        long_rt = []
        for video in self._videos:
            long_gt += list(video.gt)
            long_rt += list(video.temp)
        long_rt = np.array(long_rt)

        kmeans_labels = np.asarray(kmean.labels_).copy()
        time2label = {}
        for label in np.unique(kmeans_labels):
            cluster_mask = kmeans_labels == label
            r_time = np.mean(long_rt[self._total_fg_mask][cluster_mask])
            time2label[r_time] = label

        np.random.seed(opt.seed)
        shuffle_labels = np.arange(len(time2label))
        np.random.shuffle(shuffle_labels)
        logger.debug(['time ordered labels', 'shuffled labels'][opt.shuffle_order])
        for time_idx, sorted_time in enumerate(sorted(time2label)):
            label = time2label[sorted_time]
            if opt.shuffle_order:
                # logger.debug('shuffled labels')
                kmeans_labels[kmean.labels_ == label] = shuffle_labels[time_idx]
            else:
                # logger.debug('time ordered labels')
                kmeans_labels[kmean.labels_ == label] = time_idx
                shuffle_labels = np.arange(len(time2label))

        labels_with_bg = np.ones(len(self._total_fg_mask)) * -1

        if opt.shuffle_order and opt.kmeans_shuffle:
            # use pure output of kmeans algorithm
            logger.debug('kmeans random labels')
            labels_with_bg[self._total_fg_mask] = kmean.labels_
            shuffle_labels = [value for (key, value) in sorted(time2label.items(), key=lambda x: x[0])]
        else:
            # use predefined by time order or numpy shuffling labels for kmeans clustering
            logger.debug('assignment: %s' % ['ordered', 'shuffled'][opt.shuffle_order])
            labels_with_bg[self._total_fg_mask] = kmeans_labels

        logger.debug('Order of labels: %s %s' % (str(shuffle_labels), str(sorted(time2label))))
        accuracy.predicted_labels = labels_with_bg
        accuracy.gt_labels = long_gt
        old_mof, total_fr = accuracy.mof()
        self._gt2label = accuracy._gt2cluster
        for key, val in self._gt2label.items():
            try:
                self._label2gt[val[0]] = key
            except IndexError:
                pass


        logger.debug('MoF val: ' + str(accuracy.mof_val()))
        logger.debug('old MoF val: ' + str(float(old_mof) / total_fr))

        ########################################################################
        # VISUALISATION
        if opt.vis and opt.vis_mode != 'segm':
            self.vis = Visual(mode=opt.vis_mode, save=True, svg=False)
            postfix = ['', '+rt.cc.'][opt.rt_cl_concat]
            self.vis.fit(self._embedded_feat, long_gt, 'gt_', reset=False)
            self.vis.color(long_rt, 'time_')
            self.vis.color(kmean.labels_, 'kmean_%s' % postfix)
            # vis.fit(self._embedded_feat, long_rt, 'time_')
            # vis.fit(self._embedded_feat, kmean.labels_, 'kmean_%s' % postfix)
        ########################################################################

        logger.debug('Update video z for videos before GMM fitting')
        labels_with_bg[labels_with_bg == self._K] = -1
        for video in self._videos:
            video.update_z(labels_with_bg[video.global_range])
            # video._z = kmean.labels_[video.global_range]

        for video in self._videos:
            video.segmentation['cl'] = (video._z, self._label2gt)

    def _count_subact(self):
        self._subact_counter = np.zeros(self._K)
        for video in self._videos:
            self._subact_counter += video.a

    def ordering_sampler(self):
        """Sampling ordering for each video via Mallow model"""
        logger.debug('.')
        self._inv_count_stat = np.zeros(self._K - 1)
        bg_total = 0
        # pr_orders = []
        for video_idx, video in enumerate(self._videos):
            video.iter = self.iter

            cur_order = video.ordering_sampler(mallow_model=self._mallow)
            # cur_order = video.viterbi_top_perm()

            # if cur_order not in pr_orders:
            #     logger.debug(str(cur_order))
            #     pr_orders.append(cur_order)
            self._inv_count_stat += video.inv_count_v
            # logger.debug('background: %d' % int(video.fg_mask.size - np.sum(video.fg_mask)))
            bg_total += int(video.fg_mask.size - np.sum(video.fg_mask))
        logger.debug('total background: %d' % bg_total)
        logger.debug('inv_count_vec: %s' % str(self._inv_count_stat))

    def rho_sampling(self):
        """Sampling of parameters for Mallow Model using slice sampling"""
        logger.debug('rho sampling')
        # self._mallow.rho = []
        mallow_rho = []
        inv_pdf = lambda x: -1. / self._mallow.logpdf(x)
        for k in range(self._K - 1):
            # logger.debug('rho sampling k: %d' % k)
            self._mallow.set_sample_params(sum_inv_vals=self._inv_count_stat[k],
                                           k=k, N=len(self._videos))
            sample = slice_sampling(burnin=10, x_init=self._mallow.rho[k],
                                    logpdf=inv_pdf)
            mallow_rho.append(sample)
        self._mallow.rho = mallow_rho
        logger.debug(str(['%.4f' % i for i in self._mallow.rho]))

    @timing
    def viterbi_ordering(self):
        logger.debug('.')
        self._inv_count_stat = np.zeros(self._K - 1)
        bg_total = 0
        for video_idx, video in enumerate(self._videos):
            if video_idx % 20 == 0:
                logger.debug('%d / %d' % (video_idx, len(self._videos)))
            video.iter = self.iter
            video.viterbi_ordering(mallow_model=self._mallow)
            self._inv_count_stat += video.inv_count_v
            # logger.debug('background: %d' % int(video.fg_mask.size - np.sum(video.fg_mask)))
            bg_total += int(video.fg_mask.size - np.sum(video.fg_mask))
        logger.debug('total background: %d' % bg_total)
        logger.debug(str(self._inv_count_stat))

    @timing
    def viterbi_decoding(self):
        logger.debug('.')
        self._count_subact()
        pr_orders = []
        for video_idx, video in enumerate(self._videos):
            if video_idx % 20 == 0:
                logger.debug('%d / %d' % (video_idx, len(self._videos)))
                self._count_subact()
                logger.debug(str(self._subact_counter))
            video.viterbi()
            cur_order = list(video._pi)
            if cur_order not in pr_orders:
                logger.debug(str(cur_order))
                pr_orders.append(cur_order)
        self._count_subact()
        logger.debug(str(self._subact_counter))

    def _action_presence_counter(self):
        """Count how many times each action occurs within video collection. Lens model."""
        presence = np.zeros(self._K)
        for video in self._videos:
            presence += np.asarray(video.a) > 1
        return presence

    def viterbi_alex_decoding(self):
        logger.debug('.')
        self._count_subact()
        len_model = np.asarray(self._subact_counter) / self._action_presence_counter()
        for video_idx, video in enumerate(self._videos):
            if video_idx % 20 == 0:
                logger.debug('%d / %d' % (video_idx, len(self._videos)))
                self._count_subact()
                logger.debug(str(self._subact_counter))
            video.viterbi_alex(len_model)
        self._count_subact()
        logger.debug(str(self._subact_counter))

    @timing
    def subactivity_sampler(self):
        """Sampling of subactivities for each video from multinomial distribution"""
        logger.debug('.')
        self._count_subact()
        for idx, video in enumerate(self._videos):
            if idx % 20 == 0:
                logger.debug('%d / %d' % (idx, len(self._videos)))
                logger.debug(str(self._subact_counter))
            temp_sub_counter = self._subact_counter - video.a
            a, _ = video.subactivity_sampler(self._subact_counter)
            self._subact_counter = temp_sub_counter + a
        logger.debug(str(self._subact_counter))

    def without_temp_emed(self):
        logger.debug('No temporal embedding')
        self._embedded_feat = self._features.copy()


    @timing
    def accuracy_corpus(self, prefix=''):
        """Calculate metrics as well with previous correspondences between
        gt labels and output labels"""
        accuracy = Accuracy()
        f1_score = F1Score(K=self._K, n_videos=len(self._videos))
        long_gt = []
        long_pr = []
        long_rel_time = []
        self.return_stat = {}

        for video in self._videos:
            long_gt += list(video.gt)
            long_pr += list(video._z)
            try:
                long_rel_time += list(video.temp)
            except AttributeError:
                pass
                # logger.debug('no poses')
        accuracy.gt_labels = long_gt
        accuracy.predicted_labels = long_pr
        if opt.bg:
            # enforce bg class to be bg class
            accuracy.exclude[-1] = [-1]

        old_mof, total_fr = accuracy.mof(old_gt2label=self._gt2label)
        self._gt2label = accuracy._gt2cluster
        self._label2gt = {}
        for key, val in self._gt2label.items():
            try:
                self._label2gt[val[0]] = key
            except IndexError:
                pass
        acc_cur = accuracy.mof_val()
        logger.debug('%sAction: %s' % (prefix, self._subaction))
        logger.debug('%sMoF val: ' % prefix + str(acc_cur))
        logger.debug('%sprevious dic -> MoF val: ' % prefix + str(float(old_mof) / total_fr))

        accuracy.mof_classes()
        accuracy.iou_classes()

        self.return_stat = accuracy.stat()

        f1_score.set_gt(long_gt)
        f1_score.set_pr(long_pr)
        f1_score.set_gt2pr(self._gt2label)
        if opt.bg:
            f1_score.set_exclude(-1)
        f1_score.f1()

        for key, val in f1_score.stat().items():
            self.return_stat[key] = val

        for video in self._videos:
            video.segmentation[video.iter] = (video._z, self._label2gt)

        if opt.vis:
            # VISUALISATION

            # gt_plot_iter = [[0, 1], [0]][self.iter != 0]
            if opt.vis_mode != 'segm':
                long_pr = [self._label2gt[i] for i in long_pr]

                if self.vis is None:
                    self.vis = Visual(mode=opt.vis_mode, save=True, reduce=None)
                    self.vis.fit(self._embedded_feat, long_pr, 'iter_%d' % self.iter)
                else:
                    reset = prefix == 'final'
                    self.vis.color(labels=long_pr, prefix='iter_%d' % self.iter, reset=reset)
            else:
                ####################################################################
                # segmentation visualisation
                if prefix == 'final':
                    colors = {}
                    cmap = plt.get_cmap('tab20')
                    for label_idx, label in enumerate(np.unique(long_gt)):
                        if label == -1:
                            colors[label] = (0, 0, 0)
                        else:
                            # colors[label] = (np.random.rand(), np.random.rand(), np.random.rand())
                            colors[label] = cmap(label_idx / len(np.unique(long_gt)))

                    dir_check(os.path.join(opt.dataset_root, 'plots'))
                    dir_check(os.path.join(opt.dataset_root, 'plots', opt.subaction))
                    fold_path = os.path.join(opt.dataset_root, 'plots', opt.subaction, 'segmentation')
                    dir_check(fold_path)
                    for video in self._videos:
                        path = os.path.join(fold_path, video.name + '.png')
                        name = video.name.split('_')
                        name = '_'.join(name[-2:])
                        plot_segm(path, video.segmentation, colors, name=name)
                ####################################################################

        return accuracy.frames()

    def resume_segmentation(self):
        logger.debug('resume precomputed segmentation')
        for video in self._videos:
            video.iter = self.iter
            video.resume()
        self._count_subact()

    def save_embed_feat(self):
        dir_check(ops.join(opt.data, 'embed'))
        dir_check(ops.join(opt.data, 'embed', opt.subaction))
        for video in self._videos:
            video_features = self._embedded_feat[video.global_range]
            feat_name = opt.resume_str + '_%s' % video.name
            np.savetxt(ops.join(opt.data, 'embed', opt.subaction, feat_name), video_features)



