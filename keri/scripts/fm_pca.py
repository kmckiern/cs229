#!/bin/env python

"""
for a feature matrix, do PCA and write results to dataframe
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
        default='cand_parse_all_fresh_2014_feat_matrix_trim_normed.pkl')
parser.add_argument('--pref', type=str, help='output file directory preface',
        default='../../data/features/')
parser.add_argument('--pca', type=int, help='number of PCA '
        'components', default=0)
args = parser.parse_args()

def main():
    # get data frame
    file_name = args.pref + args.cf
    df = pd.read_pickle(file_name)

    # do PCA
    nc = args.pca
    if nc > 0:
        from sklearn.decomposition import PCA
        pca = PCA(n_components = nc)
        pca.fit(df.T)
        PCs = pca.components_
        cols = ['PC ' + str(i) for i in range(PCs.shape[0])]
        pc_df = pd.DataFrame(PCs.T, columns=cols)
        of = file_name.replace('.pkl', '_pc' + str(nc) + '.pkl')
        pc_df.to_pickle(of)

if __name__ == '__main__':
    main()
