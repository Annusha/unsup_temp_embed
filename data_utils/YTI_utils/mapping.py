#!/usr/bin/env python

"""Some preprocessing to get proper format of gt.
Without label index = 0, due to the fact that in Breakfast dataset there is SIL
label, from which I got rid of during training. To simplify everything I just
won't use 0 as index for this (YTI) dataset.
-1 - background index
"""

__author__ = 'Anna Kukleva'
__date__ = 'September 2018'


import os
import re

from ute.utils.util_functions import dir_check
from ute.utils.arg_pars import opt

actions = ['coffee', 'changing_tire', 'cpr', 'jump_car', 'repot']
gt_folder = '/media/data/kukleva/lab/YTInstructions/segmentation_gt_dt'
dir_check(opt.gt)

label2idx = {}
idx2label = {}

label2idx['bg'] = -1
idx2label[-1] = 'bg'

videos = {}

for root, dirs, files in os.walk(gt_folder):
    for filename in files:
        segmentation = []
        with open(os.path.join(root, filename), 'r') as f:
            for line in f:
                match = re.match(r'(\d*)-(\d*)\s*(\w*)', line)
                start = int(match.group(1))
                end = int(match.group(2))
                label = match.group(3)
                segmentation += [label] * (end - start + 1)
                if label not in label2idx:
                    label_idx = len(label2idx) - 1
                    label2idx[label] = label_idx
                    idx2label[label_idx] = label
        match = re.match(r'(\w*)\.\w*', filename)
        gt_name = match.group(1)
        with open(os.path.join(opt.gt, gt_name), 'w') as f:
            for label in segmentation:
                f.write('%s\n' % label)
        videos[filename] = segmentation

dir_check(os.path.join(opt.gt, 'mapping'))
with open(os.path.join(opt.gt, 'mapping', 'mapping_idxlabel.txt'), 'w') as f:
    for idx in sorted(idx2label):
        f.write('%d %s\n' % (idx, idx2label[idx]))




