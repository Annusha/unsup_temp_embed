#!/usr/bin/env python

"""Eval level activity
"""

__author__ = 'Anna Kukleva'
__date__ = 'January 2019'

import os

from utils.arg_pars import opt
import FS_utils.update_argpars as fs_utils


fs_utils.update()

actions = ['add_oil',
           'add_pepper',
           'mix_dressing',
           'mix_ingredients',
           'cut_into_pieces',
           'place_into_bowl',
           'peel_cucumber',
           'serve_salad',
           'dress_salad']

high = {}
high['action_start'] = ['action_start']
high['cut_and_mix_eingredients'] = ['peel_cucumber',
                                    'cut_cucumber',
                                    'place_cucumber_into_bowl',
                                    'cut_tomato',
                                    'place_tomato_into_bowl',
                                    'cut_cheese',
                                    'place_cheese_into_bowl',
                                    'cut_lettuce',
                                    'place_lettuce_into_bowl',
                                    'mix_ingredients']
high['prepare_dressing'] = ['add_oil',
                            'add_vinegar',
                            'add_salt',
                            'add_pepper',
                            'mix_dressing']
high['serve_salad'] = ['serve_salad_onto_plate',
                       'add_dressing']
high['action_end'] = ['action_end']

label2idx = {}
idx2label = {}
path = os.path.join(opt.dataset_root, opt.gt, 'mapping', 'map_high.txt')
with open(path, 'w') as f:
    for idx, (high_act, mid_acts) in enumerate(high.items()):
        for mid_act in mid_acts:
            f.write('%d %s\n' % (idx, mid_act))
