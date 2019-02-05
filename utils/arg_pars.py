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
actions = ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'friedegg', 'salat', 'pancake']
parser.add_argument('--subaction', default='juice',
                    help='measure accuracy for different subactivities scrambledegg')
parser.add_argument('--all', default=False, type=bool,
                    help='to process in pipeline all subactions of the corresponding '
                         'dataset')
parser.add_argument('--dataset', default='bf',
                    help='Breakfast dataset (bf) or YouTube Instructional (yti)'
                         'or 50 Salads (fs)')
parser.add_argument('--data_type', default=1, type=int,
                    help='valid just for Breakfast dataset and 50 Salads'
                         '0: kinetics - features from the stream network'
                         '1: data - normalized features'
                         '2: s1 - features without normalization'
                         '3: videos'
                         ''
                         '0: kinetics'
                         '2: s1 - dense trajectories wo normalization')
parser.add_argument('--f_norm', default=False, type=bool,
                    help='normalization of the features')


parser.add_argument('--dataset_root', default='',
                    help='root folder for dataset:'
                         'Breakfast / YTInstructions')
parser.add_argument('--data', default='',
                    help='direct path to your data features')
parser.add_argument('--gt', default='groundTruth',
                    help='folder with ground truth labels')
parser.add_argument('--high', default=False,
                    help='switch between different levels of labels')
parser.add_argument('--feature_dim', default=64,
                    help='feature dimensionality')
parser.add_argument('--ext', default='',
                    help='extension of the feature files')


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
parser.add_argument('--epochs', default=30, type=int,
                    help='number of epochs for training embedding')
parser.add_argument('--gt_training', default=False, type=bool,
                    help='training embedding either with gt labels '
                         'or with labels gotten from the temporal model')


###########################################
# hyperparams parameters for TCN embeddings

parser.add_argument('--levels', type=int, default=1,
                    help='number of levels (default: 1)')
parser.add_argument('--ksize', type=int, default=2,
                    help='kernel size (default: 7)')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--tcn_len', type=int, default=10,
                    help='length of the history')


###########################################
# probabilistic parameters
parser.add_argument('--gmm', default=1, type=int,
                    help='number of components for gaussians')
parser.add_argument('--gmms', default='one',
                    help='number of gmm for the video collection: many/one')
parser.add_argument('--reg_cov', default=1e-4, type=float)
parser.add_argument('--ordering', default=False,
                    help='apply Mallow model for incorporate ordering')
parser.add_argument('--shuffle_order', default=False, type=bool,
                    help='shuffle or order wrt relative time cluster labels after clustering')
parser.add_argument('--kmeans_shuffle', default=False, type=bool,
                    help='auto shuffle after kmeans or'
                         'shuffle enforced numpy with given seed')


###########################################
# vae
parser.add_argument('--rt_concat', default=0, type=int,
                    help='additional dimensionality for vae')
parser.add_argument('--label', default=False, type=bool,
                    help='features for training embedding is + concat with '
                         'uniform label')
parser.add_argument('--concat', default=1, type=int,
                    help='how much consecutive features to concatenate')

###########################################
# bg
parser.add_argument('--bg', default=False, type=bool,
                    help='if we need to apply part for modeling background')
parser.add_argument('--bg_trh', default=55, type=int)

###########################################
# viterbi
parser.add_argument('--viterbi', default=True, type=bool)

###########################################
# save
parser.add_argument('--save_model', default=True, type=bool,
                    help='save embedding model after training')
parser.add_argument('--save_embed_feat', default=False,
                    help='save features after embedding trained on gt')
parser.add_argument('--save_likelihood', default=False, type=bool)
parser.add_argument('--resume_segmentation', default=False, type=bool)
parser.add_argument('--resume', default=True, type=bool,
                    help='load model for embeddings, if positive then it is number of '
                         'epoch which should be loaded')
parser.add_argument('--resume_str',
                    # for Breakfast dataset
                    # default='!norm.!conc._%s_mlp_!pose_full_vae0_time10.0_epochs90_embed20_n2_ordering_gmm1_one_!gt_lr0.001_lr_!zeros_b0_v1_l0_c1_',
                    # default='grid.vit._%s_mlp_!pose_full_vae1_time10.0_epochs90_embed20_n2_ordering_gmm1_one_!gt_lr0.001_lr_zeros_b0_v1_l0_c1_',
                    # norm.!conc.
                    default='fixed.order._%s_mlp_!pose_full_vae0_time10.0_epochs60_embed20_n1_!ordering_gmm1_one_!gt_lr0.0001_lr_zeros_b0_v1_l0_c1_',
                    # default='norm.conc._%s_mlp_!pose_full_vae1_time10.0_epochs60_embed20_n1_ordering_gmm1_one_!gt_lr0.0001_lr_!zeros_b0_v1_l0_c1_',

                    # for YouTube Instructions dataset
                    # default='yti.(200,90,-3)_%s_mlp_!pose_full_vae0_time10.0_epochs90_embed200_n4_!ordering_gmm1_one_!gt_lr0.001_lr_zeros_b1_v1_l0_c1_',

                    # for 50 salads dataset
                    # default='50s.gs._%s_!bg_cc1_data2_fs_dim30_ep30_gmm1_!gt_!l_lr0.001_mlp_!mal_size0_+d0_vit_',
                    # default='full._%s_!bg_cc1_data2_fs_dim30_ep30_gmm1_!gt_!l_lr0.001_mlp_!mal_size0_+d0_vit_',

                    # for Breakfast rank model
                    # default='rank._%s_rank_!pose_full_vae0_time10.0_epochs30_embed30_n2_!ordering_gmm1_one_!gt_lr1e-06_lr_!zeros_b0_v1_l0_c1_b0_',
                    # default='rank._%s_rank_!pose_full_vae0_time10.0_epochs30_embed30_n2_!ordering_gmm1_one_!gt_lr1e-06_lr_zeros_b0_v1_l0_c1_b96_',
                    )


###########################################
# additional
parser.add_argument('--reduced', default=15, type=int,
                    help='check smth using only ~15 videos')
parser.add_argument('--grid_search', default=False, type=bool,
                    help='grid search for optimal parameters')
parser.add_argument('--vis', default=True, type=bool,
                    help='save visualisation of embeddings')
parser.add_argument('--vis_mode', default='pca',
                    help='pca / tsne')
parser.add_argument('--model_name', default='mlp',
                    help='mlp / tcn')
parser.add_argument('--test_set', default=False, type=bool,
                    help='check if the network if overfitted or not')
parser.add_argument('--prefix', default='vis.',
                    help='prefix for log file')


###########################################
# additional temporary parameters
parser.add_argument('--rt_cl_concat', type=bool, default=False,
                    help='concatenation with relative time label before clustering of embedded features')
parser.add_argument('--gaussian_cl', type=bool, default=False)


###########################################
# logs
parser.add_argument('--log', default='DEBUG',
                    help='DEBUG | INFO | WARNING | ERROR | CRITICAL ')
parser.add_argument('--log_str', default='',
                    help='unify all savings')


opt = parser.parse_args()

