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
args, unknown = parser.parse_known_args()

def main():
    # read in data
    data = pd.read_pickle(args.fm)
    cols = ['name', 'state', 'CID', 'party', 'dwn0', 'dwn1']
    cands = pd.read_csv(args.cf, sep='\t', names=cols)
    cids = np.array(cands['CID'])
    dwn0 = np.array(cands['dwn0'])
    dwn1 = np.array(cands['dwn1'])

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

    IPython.embed()

if __name__ == '__main__':
    main()
