#!/bin/env python

"""
for a list of candidates, generate a matrix of features
"""

import os
import argparse
import pandas as pd
import numpy as np
import os_cid_call

parser = argparse.ArgumentParser(description='get financial information for an'
        ' input candidate')
parser.add_argument('--cf', type=str, help='file of candidates and CIDs',
        default='../../joe/out/split/xaa')
parser.add_argument('--yr', type=str, help='query year', default='2014')
parser.add_argument('--ak', type=str, help='api key',
        default='49f6c4fb24c044acdfcd42b5d559d343')
parser.add_argument('--categories', type=str, help='sector code database file',
        default='../../data/candidates/CRP_Categories.txt')
parser.add_argument('--pref', type=str, help='output file directory preface',
        default='../../data/features/')
parser.add_argument('--norm', action='store_true', help='normalize fin data',
        default=False)
args = parser.parse_args()

def main():
    # get candidate 
    cols = ['f name', 'l name', 'state', 'party', 'CID', 'dwn0', 'dwn1']
    cands = pd.read_csv(args.cf, sep='\t', names=cols)
    cids = list(cands['CID'])
    nc = cands.shape[0]

    dwn0 = cands['dwn0']
    dwn1 = cands['dwn1']
    scores = np.zeros((len(dwn0), 2))
    scores[:,0] = dwn0
    scores[:,1] = dwn1

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
    if args.norm:
        df.to_pickle(args.pref + from_str + '_' + str(args.yr) + '_feat_matrix.pkl')
    else:
        df.to_pickle(args.pref + from_str + '_' + str(args.yr) +
                '_feat_matrix_unnormed.pkl')
    
    # also dwn scores to a data frame pkl file
    df_y = pd.DataFrame(scores, columns=cols[-2:], index=cids)
    df_y.to_pickle(args.pref + from_str + '_' + str(args.yr) + '_scores.pkl')

if __name__ == '__main__':
    main()
