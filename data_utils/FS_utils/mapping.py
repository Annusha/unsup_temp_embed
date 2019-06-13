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

from ute.utils.util_functions import dir_check
from ute.utils.arg_pars import opt
import data_utils.FS_utils.update_argpars as fs_utils

actions = ['-1.', '-2.']
fs_utils.update()
dir_check(opt.gt)

label2idx = {}
idx2label = {}

videos = {}

for filename in os.listdir(opt.gt):
    with open(os.path.join(opt.gt, filename), 'r') as f:
        for line in f:
            line = line.strip()
            if label2idx.get(line, -1) == -1:
                idx = len(idx2label)
                label2idx[line] = idx
                idx2label[idx] = line

dir_check(os.path.join(opt.gt, 'mapping'))
with open(os.path.join(opt.gt, 'mapping', 'mapping.txt'), 'w') as f:
    for idx in sorted(idx2label):
        f.write('%d %s\n' % (idx, idx2label[idx]))




