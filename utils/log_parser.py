#!/usr/bin/env python

"""Log parser"""

__author__ = 'Anna Kukleva'
__date__ = 'September 2018'


import numpy as np
import os
import pandas as pd
from collections import defaultdict
import re

from utils.arg_pars import opt


def params_parser(path):

    params = defaultdict(list)
    set_idx = 0

    def reset(line):
        nonlocal params, set_idx
        params = defaultdict(list)
        set_idx += 1

        search = re.search(r'dim:\s*(\d*)\s*epochs:\s*(\d*)\s*,\s*lr:\s*(\d*.\d*e-\d*)\s*', line)
        params['dim'] = int(search.group(1))
        params['epochs'] = int(search.group(2))
        params['lr'] = float(search.group(3))
        params['idx'] = set_idx

    with open(path, 'r') as f:
        frames = []
        for line in f:
            if 'wrap - <function pose_training' in line:
                yield params

            if 'SET' in line:
                print('%d : %s' % (set_idx, line))
                reset(line)

            if 'clustering - MoF val' in line:
                params['cl_mof'].append(float(line.split()[-1]))

            if 'accuracy_corpus - MoF val' in line:
                params['mof'].append(float(line.split()[-1]))
                params['frames'].append(frames[-1])

            if 'accuracy_corpus - pure vit MoF val' in line:
                params['!o_mof'].append(float(line.split()[-1]))
                params['!o_frames'].append(frames[-1])

            if 'mof_val - frames true:' in line:
                search = re.search(r'frames true:\s*(\d*)\s*frames overall :\s*(\d*)', line)
                frames.append(int(search.group(1)))

            if 'average class mof:' in line:
                params['av_mof'].append(float(line.split()[-1]))


def params_parser_tcn(path):

    params = defaultdict(list)
    set_idx = 0

    def reset(line):
        nonlocal params, set_idx
        params = defaultdict(list)
        set_idx += 1

        search = re.search(r'dim:\s*(\d*)\s*lr:\s*(\d*.\d*e-\d*)\s*ksize\s*(\d*)\s*levels\s*(\d*)\s*\s*dropout\s*(\d*.\d*)\s*epochs\s*(\d*)\s*', line)
        params['dim'] = int(search.group(1))
        params['lr'] = float(search.group(2))
        params['ksize'] = int(search.group(3))
        params['levels'] = int(search.group(4))
        params['dropout'] = float(search.group(5))
        params['epochs'] = int(search.group(6))
        params['idx'] = set_idx

    with open(path, 'r') as f:
        frames = []
        for line in f:
            if 'wrap - <function temp_embed' in line:
                yield params

            if 'SET' in line:
                print('%d : %s' % (set_idx, line))
                reset(line)

            if 'clustering - MoF val' in line:
                params['cl_mof'].append(float(line.split()[-1]))

            if 'accuracy_corpus - MoF val' in line:
                params['mof'].append(float(line.split()[-1]))
                params['frames'].append(frames[-1])

            if 'accuracy_corpus - finalMoF val:' in line:
                params['final_mof'].append(float(line.split()[-1]))
                params['frames'].append(frames[-1])

            if 'mof_val - frames true:' in line:
                search = re.search(r'frames true:\s*(\d*)\s*frames overall :\s*(\d*)', line)
                frames.append(int(search.group(1)))


def table_joiner(prefix, path):
    if opt.dataset == 'bf':
        table_path = '/media/data/kukleva/lab/Breakfast/tables'
    if opt.dataset == 'yti':
        table_path = '/media/data/kukleva/lab/YTInstructions/tables'
    all_subactions = {}

    filelist = os.listdir(path)
    for filename in filelist:
        if filename.startswith(prefix):
            params = defaultdict(list)
            subaction = ''
            with open(os.path.join(path, filename), 'r') as f:
                for line in f:
                    # subaction
                    if 'update - subaction:' in line:
                        subaction = line.strip().split()[-1]
                    # range
                    if 'Order of labels:' in line:
                        cl_range = [int(100 * float(i)) for i in line.strip().split('[')[-1].split(']')[0].split(',')]
                        params['cl_range'] = (cl_range[0], cl_range[-1])
                    # MoF
                    if 'accuracy_corpus - MoF val:' in line or \
                            'accuracy_corpus - finalMoF val:' in line:
                        mof = int(float(line.strip().split()[-1]) * 100)
                        params['mof'].append(mof)
                    # frames
                    if 'mof_val - frames true:' in line:
                        search = re.search(r'frames true:\s*(\d*)\s*frames overall :\s*(\d*)', line)
                        params['frames'] = [int(search.group(1))]  # save only last frames
            all_subactions[subaction] = params
    subactions = ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'friedegg', 'salat', 'pancake']
    data = {'data': []}
    for subaction in subactions:
        keys = ['cl_range', 'mof']
        for key in keys:
            data['data'] += [all_subactions[subaction][key]]
        data['data'] += all_subactions[subaction]['frames']
        data['data'] += [' ']

    df = pd.DataFrame(data)
    print(df)
    name = prefix + path.split('/')[-1] + '.csv'
    df.to_csv(os.path.join(table_path, name))


def all_subactions_parser(prefix, path):
    if opt.dataset == 'bf':
        table_path = '/media/data/kukleva/lab/Breakfast/tables'
    if opt.dataset == 'yti':
        table_path = '/media/data/kukleva/lab/YTInstructions/tables'
    subactions = ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'friedegg', 'salat', 'pancake']

    all_subactions = {}

    filelist = os.listdir(path)
    for filename in filelist:
        if filename.startswith(prefix):
            with open(os.path.join(path, filename), 'r') as f:
                for line in f:
                    # subaction
                    if 'update_opt_str - subaction:' in line:
                        subaction = line.strip().split()[-1]
                        params = defaultdict(list)
                    # range
                    if 'Order of labels:' in line:
                        cl_range = [int(100 * float(i)) for i in line.strip().split('[')[-1].split(']')[0].split(',')]
                        params['cl_range'] = (cl_range[0], cl_range[-1])
                    # MoF
                    if 'accuracy_corpus - MoF val:' in line or \
                            'accuracy_corpus - finalMoF val:' in line:
                        mof = int(float(line.strip().split()[-1]) * 100)
                        params['mof'].append(mof)
                    # frames
                    if 'mof_val - frames true:' in line:
                        search = re.search(r'frames true:\s*(\d*)\s*frames overall :\s*(\d*)', line)
                        params['frames'] = [int(search.group(1))]  # save only last frames

                    if 'util_functions.py - wrap - <function temp_embed' in line:
                        all_subactions[subaction] = params
            data = {'data': []}

            for subaction in subactions:
                keys = ['cl_range', 'mof']
                for key in keys:
                    data['data'] += [all_subactions[subaction][key]]
                data['data'] += all_subactions[subaction]['frames']
                data['data'] += [' ']

            df = pd.DataFrame(data)
            print(df)
            name = filename + path.split('/')[-1] + '.csv'
            df.to_csv(os.path.join(table_path, name))




def create_table_params(path, prefix=''):
    if opt.dataset == 'bf':
        table_path = '/media/data/kukleva/lab/Breakfast/tables'
    if opt.dataset == 'yti':
        table_path = '/media/data/kukleva/lab/YTInstructions/tables'
    data = defaultdict(list)

    for params in params_parser_tcn(path):
        if not params:
            continue

        # name = (params['dim'], params['epochs'], params['lr'])
        name = (params['dim'], params['lr'], params['ksize'], params['levels'], params['dropout'], params['epochs'])
        if not name[0]:
            name = 'set'
        # data[name].append(params['loss'][-1])
        data[name].append(int(100 * params['mof'][0]))
        data[name].append(int(100 * params['final_mof'][0]))

        # mof = [int(i * 100) for i in params['mof']]
        # mof = [int(100 * params['mof'][0]), int(100 * params['!o_mof'][0])]  # for pure vit
        # data[name].append(mof)

        # av_mof = [int(i * 100) for i in params['av_mof']]
        # data[name].append(av_mof)

        # frames = [params['frames'][0], params['!o_frames'][0]]
        # data[name].append(frames)
        # data[name].append(frames[-1])
        data[name].append(params['frames'])
        data[name].append(params['frames'][1])


    df = pd.DataFrame(data)
    # decimals = pd.Series([2], index=['cl_mof'])
    # df.round(decimals)
    print(df)
    name = prefix + path.split('/')[-1] + '.csv'
    df.to_csv(os.path.join(table_path, name))


def yti_parser(p):
    with open(p) as f:
        action = ''
        mof = 0
        mof_bg = 0
        trh = 0
        f1 = 0
        for line in f:
            if 'accuracy_corpus - Action:' in line:
                action = line.split()[-1]
            if 'accuracy_corpus - MoF val:' in line:
                mof = line.split()[-1]
            if 'mof_classes - mof with bg:' in line:
                mof_bg = line.split()[-1]
            if 'all_actions - bg_trh' in line:
                trh = line.split()[-1]
            if 'f1_score.py - f1 - f1 score:' in line:
                f1 = line.split()[-1]
            if 'util_functions.py - wrap - <function pose_training ' in line:
                print('Action: %s, %s\n'
                      'mof: %s\n'
                      'mof with bg: %s\n'
                      'f1 score: %s\n' % (action, trh, mof, mof_bg, f1))

        print('__________________________________________\n')




if __name__ == '__main__':
    # path = '/media/data/kukleva/lab/logs_debug/grid_search_mlp_tea_pipeline_mlp_full_2_20_gmm(1)_adj_lr1.0e-03_ep30(2018-09-17 21:49:14.462660)'
    # create_table(path)

    # path = '/media/data/kukleva/lab/Breakfast/logs/grid.vit._coffee_mlp_!pose_full_vae1_time10.0_epochs30_embed20_n2_ordering_gmm1_one_!gt_lr0.001_lr_zeros_b0_v1_l0_c1_pipeline(2018-10-24 23:06:18.032832)'
    #
    # path = '/media/data/kukleva/lab/Breakfast/logs/tcn.gs._coffee_!bg_cc1_data1_bf_dim40_ep10_gmm1_!gt_!l_lr0.0001_tcn_!mal_size40_+d0_vit_pipeline(2019-01-02 11:04:44.413855)'
    # create_table_params(path)

    # prefix = 'gmm.rt.cc.'
    # log_path = '/media/data/kukleva/lab/Breakfast/logs'
    # table_joiner(prefix, log_path)


    prefix = 'kmean.rt.cc.'
    log_path = '/media/data/kukleva/lab/Breakfast/logs'
    all_subactions_parser(prefix, log_path)


