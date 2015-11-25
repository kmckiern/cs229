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
parser.add_argument('--ak', type=str, help='api key')
parser.add_argument('--categories', type=str, help='sector code database file',
        default='../../data/candidates/CRP_Categories.txt')
parser.add_argument('--pref', type=str, help='output file directory preface',
        default='../../data/features/')
args = parser.parse_args()

def main():
    # get candidate info
    cols = ['name', 'state', 'CID', 'dwn0', 'dwn0']
    cands = pd.read_csv(args.cf, sep='\t', names=cols)
    cids = list(cands['CID'])
    nc = cands.shape[0]

    # to avoid module errors
    args.write_dicts = False
    args.ip = False

    # call os_cid_call.py
    for ndx, cid in enumerate(cids):
        # get feature vector and add to feature matrix
        args.cid = cid
        features, feature_lbls = os_cid_call.main(args)
        if ndx == 0:
            nf = len(features)
            fmtrx = np.zeros((nc, nf))
        fmtrx[ndx,:] = features

    # put into data frame
    df = pd.DataFrame(fmtrx, columns=feature_lbls, index=cids)
    from_str = args.cf.split('/')[-1].split('.')[0]
    df.to_pickle(args.pref + from_str + '_' + str(args.yr) + '_feat_matrix.pkl')

if __name__ == '__main__':
    main()
