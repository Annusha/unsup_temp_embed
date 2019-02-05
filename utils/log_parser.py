#!/usr/bin/env python

"""Log parser"""

__author__ = 'Anna Kukleva'
__date__ = 'September 2018'


import numpy as np
import os
import pandas as pd
from collections import defaultdict
import re

# from utils.arg_pars import opt


def params_parser(path):

    params = defaultdict(list)
    set_idx = 0

    def reset(line):
        nonlocal params, set_idx
        params = defaultdict(list)
        set_idx += 1

        search = re.search(r'dim:\s*(\d*)\s*epochs:\s*(\d*)\s*,*\s*lr:\s*(\d*.\d*e-\d*),*\s*norm:\s*(\w*)', line)
        params['dim'] = int(search.group(1))
        params['epochs'] = int(search.group(2))
        params['lr'] = float(search.group(3))
        params['norm'] = str(search.group(4))
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
    # if opt.dataset == 'bf':
    #     table_path = '/media/data/kukleva/lab/Breakfast/tables'
    # if opt.dataset == 'yti':
    #     table_path = '/media/data/kukleva/lab/YTInstructions/tables'
    table_path = '/media/data/kukleva/lab/50salads/tables'
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
    # table_path = '/media/data/kukleva/lab/Breakfast/tables'
    # table_path = '/media/data/kukleva/lab/YTInstructions/tables'
    table_path = '/media/data/kukleva/lab/50salads/tables'
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
    # if opt.dataset == 'bf':
    #     table_path = '/media/data/kukleva/lab/Breakfast/tables'
    # if opt.dataset == 'yti':
    #     table_path = '/media/data/kukleva/lab/YTInstructions/tables'
    # if opt.dataset == 'fs':
    table_path = '/media/data/kukleva/lab/50salads/tables'
    data = defaultdict(list)

    # for params in params_parser_tcn(path):
    for params in params_parser(path):
        if not params:
            continue

        name = (params['dim'], params['epochs'], params['lr'], params['norm'])
        # name = (params['dim'], params['lr'], params['ksize'], params['levels'], params['dropout'], params['epochs'])
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


def seed_parser(path):
    # just for breakfast dataset

    seeds = []
    frames = []
    total_frames = 3590899

    cur_frames = 0
    with open(path, 'r') as f:
        for line in f:
            if 'SET: seed:' in line:
                line = int(line.strip().split()[-1])
                if line not in seeds:
                    if seeds:
                        frames.append(cur_frames)
                    cur_frames = 0
                    # seeds.append(line)
                continue

            if 'accuracy_class.py - mof_val - frames true:' in line:
                search = re.search(r'frames true:\s*(\d*)\s*frames overall :\s*(\d*)', line)
                action_frames = int(search.group(1))

            if 'wrap - <function temp_embed' in line:
                cur_frames += action_frames

            if 'wrap - <function all_actions' in line:
                frames.append(cur_frames)

    return np.array(frames) / total_frames




if __name__ == '__main__':
    # create_table_params(path)

    # prefix = 'gmm.rt.cc.'
    # log_path = '/media/data/kukleva/lab/Breakfast/logs'
    # table_joiner(prefix, log_path)


    # prefix = 'kmean.rt.cc.'
    # log_path = '/media/data/kukleva/lab/Breakfast/logs'
    # all_subactions_parser(prefix, log_path)

    log_path = '/media/data/kukleva/lab/Breakfast/logs'
    frames = []
    for filename in os.listdir(log_path):
        if not filename.startswith('seeds.new.'):
            continue
        print(filename)
        frames += list(seed_parser(os.path.join(log_path, filename)))

    print(frames)
    print(np.mean(frames))
    print(len(frames))

