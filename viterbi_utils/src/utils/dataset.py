#!/usr/bin/python2.7

import numpy as np

# reads the data
#
# @base_path: path to the data directory
# @video_list: list of video names to load
# @ label2index: mapping from labels (class names) to label indices
#
# self.features[video]: the feature array of the given video (dimension x frames)
# self.action_set[video]: a set containing all occurring actions
# self.ground_truth[video]: the ground truth labels of the video
# self.input_dimension: dimension of video features
# self.n_classes: number of classes
class Dataset(object):

    def __init__(self, base_path, video_list, label2index):
        self.features = dict()
        self.action_set = dict()
        self.ground_truth = dict()
        # read features for each video
        for video in video_list:
            # video features
            self.features[video] = np.load(base_path + '/features/' + video + '.npy')
            # action set
            with open(base_path + '/transcripts/' + video + '.txt') as f:
                self.action_set[video] = set([ label2index[line] for line in f.read().split('\n')[0:-1] ])
            # ground truth
            with open(base_path + '/groundTruth/' + video + '.txt') as f:
                self.ground_truth[video] = [ label2index[line] for line in f.read().split('\n')[0:-1] ]
        # set input dimension and number of classes
        self.input_dimension = self.features.values()[0].shape[0]
        self.n_classes = len(label2index)
        self.n_frames = sum([data.shape[1] for data in self.features.values()])

    def videos(self):
        return self.features.keys()

    def length(self, video):
        return self.features[video].shape[1]
