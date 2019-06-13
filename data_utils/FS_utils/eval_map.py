#!/usr/bin/env python

"""Eval level activity
"""

__author__ = 'Anna Kukleva'
__date__ = 'January 2019'

import os

from ute.utils.arg_pars import opt
import data_utils.FS_utils.update_argpars as fs_utils


fs_utils.update()

actions = ['add_dressing',
           'add_oil',
           'add_pepper',
           'cut',
           'mix_dressing',
           'mix_ingredients',
           'peel_cucumber',
           'place',
           'serve_salad_onto_plate']

eval = {}
eval['action_start'] = ['action_start']
eval['add_dressing'] = ['add_dressing']
eval['add_oil'] = ['add_oil']
eval['add_pepper'] = ['add_pepper']
eval['cut'] = ['cut_cucumber',
               'cut_tomato',
               'cut_cheese',
               'cut_lettuce']
eval['mix_dressing'] = ['mix_dressing']
eval['mix_ingredients'] = ['mix_ingredients']
eval['peel_cucumber'] = ['peel_cucumber']
eval['place'] = ['place_cucumber_into_bowl',
                 'place_tomato_into_bowl',
                 'place_cheese_into_bowl',
                 'place_lettuce_into_bowl']
eval['serve_salad_onto_plate'] = ['serve_salad_onto_plate']
eval['null'] = ['add_salt',
                'add_vinegar']
eval['action_end'] = ['action_end']

label2idx = {}
idx2label = {}
path = os.path.join(opt.dataset_root, opt.gt, 'mapping', 'mappingeval.txt')
with open(path, 'w') as f:
    for idx, (high_act, mid_acts) in enumerate(eval.items()):
        for mid_act in mid_acts:
            f.write('%d %s\n' % (idx, mid_act))
