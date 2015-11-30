#!/bin/env python

"""
for a list of candidates, generate a matrix of features
"""

import os
import argparse
import pandas as pd
import numpy as np
import os_cid_call
import IPython

parser = argparse.ArgumentParser(description='get financial information for an'
        'input candidate')
parser.add_argument('--cf', type=str, help='file of candidates and CIDs',
        default='../../joe/out/test_update.dat')
parser.add_argument('--yr', type=str, help='query year', default='2014')
parser.add_argument('--pref', type=str, help='output file directory preface',
        default='../../data/features/')
parser.add_argument('--pca', type=int, help='number of PCA'
        'components', default=0)
args = parser.parse_args()

def main():
    # get data frame
    from_str = args.cf.split('/')[-1].split('.')[0]
    df = pd.read_pickle(args.pref + from_str + '_' + str(args.yr) + '_feat_matrix.pkl')

    nc = args.pca
    if nc > 0:
        from sklearn.decomposition import PCA
        pca = PCA(n_components = nc)
        pca.fit(df.T)
        PCs = pca.components_
        cols = ['PC ' + str(i) for i in range(PCs.shape[0])]
        pc_df = pd.DataFrame(PCs.T, columns=cols)
        pc_df.to_pickle(args.pref + from_str + '_' + str(args.yr) +
                '_feat_matrix_pc' + str(nc) + '.pkl')

if __name__ == '__main__':
    main()
