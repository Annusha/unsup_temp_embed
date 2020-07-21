#!/usr/bin/env python

"""Load all features for several classes (entire dataset) and train joined
embedding to discriminate between classes and then apply the segmentation
algorithm.
"""

__author__ = 'Anna Kukleva'
__date__ = 'July 2020'

import sys
import os
sys.path.append(os.path.abspath('.').split('full_utils')[0])


import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from ute.corpus import Corpus as GroundTruthCorpus
from ute.corpus_wrapper import CorpusWrapper
from ute.utils.util_functions import join_data, timing,  join_return_stat
from ute.utils.arg_pars import opt
from ute.utils.logging_setup import logger
from ute.models.training_embed import training, load_model
from ute.action_video import ActionVideo
from ute.eval_utils.accuracy_class import Accuracy
from ute.eval_utils.f1_score import F1Score
from ute.models import mlp
from ute.models.dataset_torch import FeatureDataset


class JoinedCorpus:
    @timing
    def __init__(self, actions, K):
        logger.debug('.')
        self._actions = actions
        self._video2idx = {}
        self._idx2video = {}
        self._gt_video2action = {}
        self._pr_video2action = {}
        self._corpuses = {}
        self._feature_list = None
        self._dataset = None
        self._features_embed = None
        self._long_gt = []
        self._pr_action_cl = None
        self._action_videos = []
        self._actions_idxs = []
        self._video_idxs = []
        self._K = K
        for action in actions:
            opt.subaction = action
            logger.debug('load %s action' % action)
            corpus = GroundTruthCorpus(K=self._K,
                                       subaction=action)
            self._corpuses[action] = corpus

            for video in corpus.get_videos():
                idx = len(self._video2idx)
                self._video2idx[video.name] = idx
        # shuffle indexes since the video loaded by actions
        shuffle_idx = np.arange(len(self._video2idx))
        np.random.shuffle(shuffle_idx)
        for idx, video_name in enumerate(self._video2idx):
            video_idx = shuffle_idx[idx]
            self._video2idx[video_name] = video_idx
            self._idx2video[video_idx] = video_name

    @timing
    def train_mlp(self):
        torch.manual_seed(opt.seed)
        self.joined_dataset(regression=True)
        dataloader = torch.utils.data.DataLoader(self._dataset,
                                                 batch_size=opt.batch_size,
                                                 shuffle=True,
                                                 num_workers=opt.num_workers)

        # opt.feature_dim = opt.feature_dim + opt.vae_dim
        # model, loss, optimizer = mlp.create_model(classes=len(self._video2idx))  # classification
        model, loss, optimizer = mlp.create_model()  # regression
        # model, loss, optimizer = mlp.create_model(joined=True, classes=len(self._video2idx))  # joined regression and classification

        if opt.load_model:
            model.load_state_dict(load_model())
            self._embedding = model
        else:
            self._embedding = training(dataloader,
                                       opt.epochs,
                                       save=opt.save_model,
                                       model=model,
                                       loss=loss,
                                       optimizer=optimizer,
                                       name='mlp')

        self._embedding = self._embedding.cpu()

        for action_idx, action in enumerate(self._actions):
            features_embed = torch.Tensor(self._corpuses[action].get_features())
            features_embed = self._embedding.embedded(features_embed).detach().numpy()

            self._features_embed = join_data(self._features_embed,
                                             features_embed,
                                             np.vstack)
            self._long_gt += list([action_idx] * features_embed.shape[0])

    @timing
    def joined_dataset(self, regression):
        logger.debug('create joined dataset')
        for action in self._actions:
            rest_features = None
            for video in self._corpuses[action].get_videos():
                video_idx = self._video2idx[video.name]
                names = np.asarray([video_idx] * video.n_frames).reshape((-1, 1))
                idxs = np.asarray(list(range(0, video.n_frames))).reshape((-1, 1))
                relative_time = np.asarray(video.temp).reshape((-1, 1))
                gt_file = relative_time.copy()
                # gt_file = names
                # relative_time *= opt.time_weight
                rest_features = join_data(rest_features,
                                          join_data(None,
                                                    (names, idxs, gt_file),
                                                    np.hstack),
                                          np.vstack)

            self._feature_list = join_data(self._feature_list,
                                           join_data(None,
                                                     (rest_features,
                                                      self._corpuses[action].get_features()),
                                                     np.hstack),
                                           np.vstack)

        self._dataset = FeatureDataset(feature_list=self._feature_list,
                                       regression=regression)


    @timing
    def preclustering_bow(self, n_clust=50):
        # cluster into clusters ffor BoW to define high level video clusters
        logger.debug('# of clusters: %d' % n_clust)
        np.random.seed(opt.seed)

        batch_size = 100 if opt.full else 30
        self.kmeans = MiniBatchKMeans(n_clusters=n_clust,
                                      random_state=opt.seed,
                                      batch_size=batch_size)

        logger.debug('Shape %d' % self._features_embed.shape[0])
        mask = np.zeros(self._features_embed.shape[0], dtype=bool)
        step = 1
        for i in range(0, mask.shape[0], step):
            mask[i] = True

        logger.debug(str(self.preclustering_bow))

        self.kmeans.fit(self._features_embed[mask])

        accuracy = Accuracy()
        accuracy.predicted_labels = self.kmeans.labels_
        accuracy.gt_labels = np.asarray(self._long_gt)[mask]

        accuracy.mof()
        logger.debug('MoF val: ' + str(accuracy.mof_val()))

    def video_level_clustering(self, n_clust):
        # form BoF and cluster them into K' clusters (high level video level labels)
        np.random.seed(opt.seed)
        long_gt = []
        long_features = None
        for action_idx, action in enumerate(self._actions):
            features = self._corpuses[action].get_features()
            features = torch.Tensor(features)
            features = self._embedding.embedded(features).detach().numpy()
            predicted_labels = self.kmeans.predict(features)
            for video_idx, video in enumerate(self._corpuses[action].get_videos()):
                video.update_z(predicted_labels[video.global_range])
                action_video = ActionVideo(name=video.name, n_clust=n_clust)

                action_video.soft_stat(features[video.global_range], self.kmeans.cluster_centers_)

                action_video.gt = [action_idx, video_idx]
                self._actions_idxs.append(action_idx)
                self._video_idxs.append(video_idx)
                self._action_videos.append(action_video)
                long_gt.append(action_idx)
                long_features = join_data(long_features,
                                          action_video.get_vec(),
                                          np.vstack)

        # video level classification
        kmeans = KMeans(n_clusters=len(self._actions) if opt.global_k_prime == 0 else opt.global_k_prime, random_state=opt.seed)
        logger.debug(str(kmeans))
        kmeans.fit(long_features)

        accuracy = Accuracy()
        accuracy.predicted_labels = kmeans.labels_
        accuracy.gt_labels = long_gt
        accuracy.mof()
        accuracy.mof_classes()
        logger.debug('MoF val: ' + str(accuracy.mof_val()))

        self._pr_action_cl = kmeans.labels_

    def segmentation(self, epochs=10, lr=1e-3, dim=30):
        long_gt = []
        long_pr = []
        pr_idx_start = 0
        return_stat_all = None
        n_videos = 0
        for pr_action_idx, pr_action in enumerate(np.unique(self._pr_action_cl)):
            logger.debug('\n\nAction # %d, label %d' % (pr_action_idx, pr_action))
            mask = self._pr_action_cl == pr_action
            pr_features = None
            pr_videos = []
            gt_actions_idxs = np.asarray(self._actions_idxs)[mask]
            video_idxs = np.asarray(self._video_idxs)[mask]
            global_start = 0
            for gt_action_idx in np.unique(gt_actions_idxs):
                mask_corpus = gt_action_idx == gt_actions_idxs
                gt_action = self._actions[gt_action_idx]

                video_idxs_corpus = video_idxs[mask_corpus]
                videos = self._corpuses[gt_action].video_byidx(video_idxs_corpus)
                feature_mask = np.zeros(videos[0].global_range.shape[0], dtype=bool)
                for video in videos:
                    feature_mask += video.global_range
                    video.global_start = global_start
                    global_start += video.n_frames

                pr_features = join_data(pr_features,
                                        self._corpuses[gt_action].get_features()[feature_mask],
                                        np.vstack)
                pr_videos += list(videos)

            total = pr_features.shape[0]
            for video in pr_videos:
                video.update_indexes(total)


            corpus_wrapper = CorpusWrapper(pr_videos, pr_features, self._K, embedding=self._embedding)
            corpus_wrapper.pipeline(epochs=epochs, lr=lr, dim=dim)
            corpus_pr, corpus_gt = corpus_wrapper.pr_gt(pr_idx_start)
            return_stat_single = corpus_wrapper.stat()
            return_stat_all = join_return_stat(return_stat_all, return_stat_single)
            pr_idx_start += corpus_wrapper._K

            long_pr += corpus_pr
            long_gt += corpus_gt

            n_videos += len(corpus_wrapper)

        # parse_return_stat(return_stat_all)

        accuracy = Accuracy()
        f1_score = F1Score(K=self._K * len(self._actions), n_videos=n_videos)
        accuracy.predicted_labels = long_pr
        accuracy.gt_labels = long_gt
        if opt.bg:
            # enforce bg class to be bg class
            accuracy.exclude[-1] = [-1]
        accuracy.mof()
        accuracy.mof_classes()
        accuracy.iou_classes()
        logger.debug('Final MoF val: ' + str(accuracy.mof_val()))

        f1_score.set_gt(long_gt)
        f1_score.set_pr(long_pr)
        f1_score.set_gt2pr(accuracy._gt2cluster)
        if opt.bg:
            f1_score.set_exclude(-1)
        f1_score.f1()

        logger.debug('Final MoF val: ' + str(accuracy.mof_val()))


def run_pipeline():
    actions = ['coffee', 'cereals', 'milk', 'tea', 'juice', 'sandwich', 'salat', 'friedegg', 'scrambledegg', 'pancake']
    # actions = ['coffee', 'cereals']


    joined_corpus = JoinedCorpus(actions=actions, K=opt.global_K)
    joined_corpus.train_mlp()

    # number of clusters for BoW
    n_clust = 50
    joined_corpus.preclustering_bow(n_clust=n_clust)
    joined_corpus.video_level_clustering(n_clust=n_clust)

    epochs = opt.local_epoch
    lr = 1e-4
    dim = opt.local_dim
    # for epoch in [30, 60, 90]:
    joined_corpus.segmentation(epochs=epochs, lr=lr, dim=dim)



if __name__ == '__main__':
    actions = ['coffee', 'cereals', 'milk', 'tea', 'juice', 'sandwich', 'salat', 'friedegg', 'scrambledegg', 'pancake']
    # actions = ['coffee', 'cereals', 'friedegg', 'scrambledegg']
    # actions = ['coffee', 'cereals', 'milk', 'tea']
    actions = ['coffee', 'cereals']

    # number of subaction clusters per group of videos
    K = opt.global_K
    # number of high level video labels
    # opt.global_k_prime = 0
    joined_corpus = JoinedCorpus(actions=actions, K=K)

    joined_corpus.train_mlp()

    # number of clusters for BoW
    n_clust = 50
    joined_corpus.preclustering_bow(n_clust=n_clust)
    joined_corpus.video_level_clustering(n_clust=n_clust)

    epochs = opt.local_epoch
    lr = 1e-4
    dim = opt.local_dim
    # for epoch in [30, 60, 90]:
    logger.debug('SET: full: %s\tordering: %s\tviterbi: %s\tK: %d\tepochs: %d\tlr: %.1e\tdim: %d' %
                 (str(opt.full), str(opt.ordering), str(opt.viterbi), K, epochs, lr, dim))
    joined_corpus.segmentation(epochs=epochs, lr=lr, dim=dim)

