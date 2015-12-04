#!/bin/env python

"""
for a list of candidates, generate a matrix of features
"""

import os
import argparse
import pandas as pd
import numpy as np
import IPython

parser = argparse.ArgumentParser(description='clean and normalize raw '
        'feature matrix')
parser.add_argument('--cf', type=str, help='file of candidates and CIDs',
        default='../../joe/out/cand_parse_all.dat')
parser.add_argument('--yr', type=str, help='query year', default='2014')
parser.add_argument('--pref', type=str, help='output file directory preface',
        default='../../data/features/')
args = parser.parse_args()

def main():
    from_str = args.cf.split('/')[-1].split('.')[0]
    df_file = args.pref + from_str + '_' + str(args.yr) \
        + '_feat_matrix_unnormed.pkl'
    df_raw = pd.read_pickle(df_file)

    # cut columns of all zeros
    df_trim = df_raw[df_raw.columns[(df_raw != 0).any()]]

    # normalize on a per candidate basis
    df_trim_norm = df_trim.div(df_trim.sum(axis=1), axis=0)

    df_file_out = df_file.replace('unnormed', 'trim_normed')
    df_trim_norm.to_pickle(df_file_out)

if __name__ == '__main__':
    main()
