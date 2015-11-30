#!/bin/env python

import IPython
import argparse
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../../joe/scripts/')
from train import *
from sklearn import cross_validation

parser = argparse.ArgumentParser(description='LOOCV over dataset')
parser.add_argument('--fm', type=str, help='feature matrix',
        default='../../data/features/cand_parse_all_2014_feat_matrix.pkl')
parser.add_argument('--cf', type=str, help='file of candidates and'
        'CIDs', default='../../joe/out/test_update.dat')
parser.add_argument('--pltd', action='store_true', help='plot dwn' 
        'distribution', default=False)
parser.add_argument('--ipnb', action='store_true', help='open ipython'
        'notebook', default=False)
args, unknown = parser.parse_known_args()

def main():
    # read in data
    data = pd.read_pickle(args.fm)
    cols = ['name', 'state', 'CID', 'party', 'DWN-0', 'DWN-1']
    cands = pd.read_csv(args.cf, sep='\t', names=cols)
    cids = np.array(cands['CID'])
    dwn0 = np.array(cands['DWN-0'])
    dwn1 = np.array(cands['DWN-1'])
    party = np.array(cands['party'])

    if args.pltd:
        import seaborn as sns 
        cm = sns.diverging_palette(20,220,n=2) #'coolwarm'
        sns.set(font_scale=.8)
        sns.set_style(style='white')
        # IPython.embed()
        party_df = sns.lmplot(x='DWN-0', y='DWN-1', hue='party', data=cands,
                fit_reg=False, palette=cm)
        party_df.savefig('../../data/out/dwn.png', dpi=400)

    # train via LOOCV
    nc = cands.shape[0]
    loo = cross_validation.LeaveOneOut(nc)
    lr = []
    svr = []
    for train_index, test_index in loo:
        X_train, X_test = np.array(data.T[cids[train_index]]).T, np.array(data.T[cids[test_index]]).T
        Y_train, Y_test = dwn0[train_index], dwn0[test_index]
        lr.append(linear_regression(X_train, Y_train, X_test, Y_test))
        svr.append(SVR(X_train, Y_train, X_test, Y_test))

    # get minimum error results
    lr = np.array(lr)
    lr_min_err = lr[lr[:,1].argmin()]
    svr = np.array(svr)
    svr_min_err = svr[svr[:,1].argmin()]

    if args.ipnb:
        IPython.embed()

if __name__ == '__main__':
    main()
