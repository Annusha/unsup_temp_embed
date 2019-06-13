#!/usr/bin/env python

"""Hyper-parameters and logging set up

opt: include all hyper-parameters
logger: unified logger for the project
"""

__all__ = ['opt']
__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import argparse
from os.path import join

parser = argparse.ArgumentParser()

###########################################
# data
actions = ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich',
           'scrambledegg', 'friedegg', 'salat', 'pancake']  # bf
actions = 'rgb'  # fs
parser.add_argument('--subaction', default='coffee',
                    help='measure accuracy for different subactivities')
parser.add_argument('--all', default=False, type=bool,
                    help='to process in pipeline all subactions of the corresponding '
                         'dataset (need additional specification in pipeline.py)')
parser.add_argument('--dataset', default='bf',
                    help='Breakfast dataset (bf) '
                         'or YouTube Instructional (yti)'
                         'or 50 Salads (fs)'
                         'run your own data (own)')
parser.add_argument('--data_type', default=0, type=int,
                    help='valid just for Breakfast dataset and 50 Salads (subaction=rgb)'
                         '0: kinetics - features from the stream network (bf)'
                         '1: data - normalized features (bf)'
                         '2: s1 - features without normalization (bf)'
                         '3: videos (bf)'
                         '4: OPN (bf)'
                         '5: videovector (bf)'
                         '6: data sabsampled with frequency 5 (bf)'
                         '7: video darwin (bf)'
                         ''
                         '0: kinetics (fs)'
                         '2: s1 - dense trajectories wo normalization (fs)')
# parser.add_argument('--subfolder', default='ascii')
parser.add_argument('--frame_frequency', default=1, type=int,
                    help='define if frequency of sampled frames and ground truth frequency are different')
parser.add_argument('--f_norm', default=True, type=bool,
                    help='feature normalization')


parser.add_argument('--dataset_root', default='',
                    help='root folder for dataset:'
                         'Breakfast / YTInstructions / 50Salads / yours')
parser.add_argument('--data', default='',
                    help='direct path to your data features')
parser.add_argument('--gt', default='groundTruth',
                    help='folder with ground truth labels')
parser.add_argument('--feature_dim', default=64,
                    help='feature dimensionality')
parser.add_argument('--ext', default='',
                    help='extension of the feature files')
parser.add_argument('--mapping_dir', default='')
parser.add_argument('--output_dir', default='')


###########################################
# hyperparams parameters for embeddings

parser.add_argument('--seed', default=0,
                    help='seed for random algorithms, everywhere')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_adj', default=True, type=bool,
                    help='will lr be multiplied by 0.1 in the middle')
parser.add_argument('--momentum', default=0.9,
                    help='momentum')
parser.add_argument('--weight_decay', default=1e-4,
                    help='regularization constant for l_2 regularizer of W')
parser.add_argument('--batch_size', default=256,
                    help='batch size for training embedding (default: 40)')
parser.add_argument('--num_workers', default=4,
                    help='number of threads for dataloading')
parser.add_argument('--embed_dim', default=20, type=int,
                    help='number of dimensions in embedded space')
parser.add_argument('--epochs', default=60, type=int,
                    help='number of epochs for training embedding')


###########################################
# bg and granularity level, dataset specific parameters
parser.add_argument('--bg', default=False, type=bool,
                    help='if we need to apply part for modeling background (yti)')
parser.add_argument('--bg_trh', default=55, type=int, help=' (yti)')
parser.add_argument('--gr_lev', default='',
                    help='switch between different levels of label granularity (fs)')

###########################################
# save
parser.add_argument('--save_model', default=True, type=bool,
                    help='save embedding model after training')
parser.add_argument('--load_embed_feat', default=False,
                    help='load features exctracted from the embedding')
parser.add_argument('--save_embed_feat', default=False,
                    help='save features after training the embedding')
parser.add_argument('--save_likelihood', default=False, type=bool)
parser.add_argument('--resume_segmentation', default=False, type=bool)
parser.add_argument('--resume', default=True, type=bool,
                    help='load model for embeddings, if positive then it is number of '
                         'epoch which should be loaded')
parser.add_argument('--load_model', default=True)
parser.add_argument('--loaded_model_name',
                    # for Breakfast dataset
                    # norm.!conc.
                    default='fixed.order._%s_mlp_!pose_full_vae0_time10.0_epochs60_embed20_n1_!ordering_gmm1_one_!gt_lr0.0001_lr_zeros_b0_v1_l0_c1_',

                    # for YouTube Instructions dataset
                    # default='yti.(200,90,-3)_%s_mlp_!pose_full_vae0_time10.0_epochs90_embed200_n4_!ordering_gmm1_one_!gt_lr0.001_lr_zeros_b1_v1_l0_c1_',

                    # for 50 salads dataset
                    # default='full._%s_!bg_cc1_data2_fs_dim30_ep30_gmm1_!gt_!l_lr0.001_mlp_!mal_size0_+d0_vit_',
                    )


###########################################
# additional
parser.add_argument('--reduced', default=0, type=int,
                    help='define how much videos to use (0 = all)')
parser.add_argument('--vis', default=False, type=bool,
                    help='save visualisation of embeddings')
parser.add_argument('--vis_mode', default='pca',
                    help='pca / tsne ')
parser.add_argument('--model_name', default='mlp',
                    help='mlp | nothing')
parser.add_argument('--test_set', default=False, type=bool,
                    help='check if the network if overfitted or not')
parser.add_argument('--device', default='cpu',
                    help='cpu | cuda')


###########################################
# logs
parser.add_argument('--prefix', default='test.',
                    help='prefix for log file')
parser.add_argument('--log', default='DEBUG',
                    help='DEBUG | INFO | WARNING | ERROR | CRITICAL ')
parser.add_argument('--log_str', default='',
                    help='unify all savings')


opt = parser.parse_args()

