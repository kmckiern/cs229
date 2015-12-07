#!/bin/env python

"""
for a list of candidates, generate a matrix of features
"""

import os
import argparse
import pandas as pd
import numpy as np
import glob
import natsort
import IPython

parser = argparse.ArgumentParser(description='clean and normalize raw '
        'feature matrix')
parser.add_argument('--cf', type=str, help='file of candidates and CIDs',
        default='../../joe/out/split/cand_parse_all_fresh.dat')
parser.add_argument('--yr', type=str, help='query year', default='2014')
parser.add_argument('--pref', type=str, help='output file directory preface',
        default='../../data/features/')
parser.add_argument('--split', type=str, help='regex if combining split '
        'pickles', default=None)
args = parser.parse_args()

def main():
    from_str = args.cf.split('/')[-1].split('.')[0]

    df_file = args.pref + from_str + '_' + str(args.yr) \
        + '_feat_matrix_unnormed.pkl'
    if args.split == None:
        df_raw = pd.read_pickle(df_file)
    else:
        # combine data matrix and score pickles
        split_files = glob.glob(args.pref + 'split_dm/*' + args.split + '*unnorm*.pkl')
        sort_files = natsort.natsorted(split_files)
        split_scores = glob.glob(args.pref + 'split_dm/*' + args.split + '*scores*.pkl')
        sort_scores = natsort.natsorted(split_scores)
        for ndx, pf in enumerate(sort_files):
            if ndx == 0:
                df_raw = pd.read_pickle(pf)
                scores = pd.read_pickle(sort_scores[ndx])
            else:
                df_raw = df_raw.append(pd.read_pickle(pf))
                scores = scores.append(pd.read_pickle(sort_scores[ndx]))
        # combine score pickles
        df_file_out = df_file.replace('_feat_matrix_unnormed', '_scores')
        scores.to_pickle(df_file_out)

    # cut columns of all zeros
    df_trim = df_raw[df_raw.columns[(df_raw != 0).any()]]

    # normalize on a per candidate basis
    df_trim_norm = df_trim.div(df_trim.sum(axis=1), axis=0)

    df_file_out = df_file.replace('unnormed', 'trim_normed')
    df_trim_norm.to_pickle(df_file_out)

if __name__ == '__main__':
    main()
