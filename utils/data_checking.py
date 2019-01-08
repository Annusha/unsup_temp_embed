#!/usr/bin/env python

"""Because I have three types of features called data, s1 and kinetics here I
checked consistency of these features between each other and with ground truth."""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import os
import numpy as np
import cv2

from utils.arg_pars import opt

'''
Create file for features which contains length 
of each feature-file (number of frames in video case)

also for different sets of the dataset there is a checking part 
to analyse if the number of feature differs of not
'''


def create_dict(path, video=False):
    # data should located in subfolder named ascii
    some_dict = {}
    for root, folders, files in os.walk(os.path.join(path, 'ascii')):
        if not files:
            continue
        for idx, filename in enumerate(files):
            if idx % 200 == 0:
                print(idx)
            if not(filename.endswith('.gz') or
                   filename.endswith('.txt') or
                   filename.endswith('.avi')):
                continue
            if not video:
                try:
                    features = np.loadtxt(os.path.join(root, filename))
                except ValueError:
                    print('Error: ', filename)
                    continue
                n_frames = features.shape[0]
            else:
                cap = cv2.VideoCapture(os.path.join(root, filename))
                if not cap.isOpened:
                    print('Cannot open video file %s' % filename)
                    continue
                n_frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
            some_dict[filename] = n_frames
    return some_dict


def write_list(path, some_dict):
    with open(os.path.join(path, 'segments', 'lens.txt'), 'w') as f:
        for key, val in some_dict.items():
            f.write('%s %d\n' % (key, val))

def checking(d1, d2, d3, d4, load=True):
    if load:
        path = '/media/data/kukleva/lab/test'
        dict_list = []
        for idx, filename in enumerate(os.listdir(path)):
            filenames_dict = {}
            with open(os.path.join(path, filename), 'r') as f:
                for line in f:
                    f_name, n_frames = line.split()
                    f_name = f_name.split('.')[0]
                    n_frames = int(n_frames)
                    filenames_dict[f_name] = n_frames
            dict_list.append(filenames_dict)
        d1, d2, d3, d4 = dict_list

    count = 0
    diff1 = 0
    diff2 = 0
    diff3 = 0
    for idx, key in enumerate(d1.keys()):
        try:
            v1 = d1[key]
            v2 = d2[key]
            v3 = d3[key]
            v4 = d4[key]
        except KeyError:
            continue

        if v1 == v2 == v3 == v4:
            continue
        if count == 0:
            diff1 = v1 - v2
            diff2 = v1 - v3
            diff3 = v1 - v4
        else:
            diff1 = -1 if diff1 - (v1 - v2) else diff1
            diff2 = -1 if diff2 - (v1 - v3) else diff2
            diff3 = -1 if diff3 - (v1 - v4) else diff3
        count += 1
    print ('Number of videos: %d\n' 
           'Video number differs: %d\n' 
           'diff1: %d\tdiff2: %d\tdiff3: %d' % (len(d4), count, diff1, diff2, diff3))


if __name__ == '__main__':

    data_dict = {}
    kinetics_dict = {}
    s1_dict = {}
    video_dict = {}

    create_dicts = False
    if create_dicts:
        print ('data features processing')
        data_dict = create_dict(opt.data)
        write_list(opt.data, data_dict)

        print ('kinetics features processing')
        kinetics_dict = create_dict(opt.kinetics)
        write_list(opt.kinetics, kinetics_dict)

        print ('s1 features processing')
        s1_dict = create_dict(opt.s1)
        write_list(opt.s1, s1_dict)

        print ('video processing')
        video_dict = create_dict(opt.video, video=True)
        write_list(opt.video, video_dict)

    print ('checking')
    checking(data_dict, kinetics_dict, s1_dict, video_dict)





