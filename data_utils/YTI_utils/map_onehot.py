#!/usr/bin/env python

""" From one hot encoding labeling to my format of gt
"""

__author__ = 'Anna Kukleva'
__date__ = 'September 2018'


import os
import re
import numpy as np

from ute.utils.arg_pars import opt
from ute.utils.util_functions import dir_check


actions = ['coffee', 'changing_tire', 'cpr', 'jump_car', 'repot']
gt_folder = '/media/data/kukleva/lab/YTInstructions/VISION_txt_annot'
dir_check(opt.gt)

label2idx = {}
idx2label = {}

videos = {}

for root, dirs, files in os.walk(gt_folder):
    for filename in files:
        segmentation = []
        with open(os.path.join(root, filename), 'r') as f:
            for line in f:
                line = line.split()
                line = list(map(lambda x: int(x), line))
                label = -1 if line[-1] == 1 else np.where(line)[0][0]
                if label != -1:
                    for action_idx, action in enumerate(actions):
                        if action in filename:
                            label += 15 * action_idx
                            break
                segmentation.append(label)
                if label not in label2idx:
                    # label_idx = len(label2idx)
                    label2idx[label] = label
                    idx2label[label] = label
        match = re.match(r'(\w*).\w*', filename)
        gt_name = match.group(1)
        with open(os.path.join(opt.gt, gt_name), 'w') as f:
            for label in segmentation:
                f.write('%d\n' % label)
        videos[filename] = segmentation


dir_check(os.path.join(opt.gt, 'mapping'))
with open(os.path.join(opt.gt, 'mapping', 'mapping.txt'), 'w') as f:
    for idx in sorted(idx2label):
        f.write('%d %d\n' % (idx, idx2label[idx]))