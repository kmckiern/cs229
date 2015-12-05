#!/bin/env python

import IPython
import argparse
import numpy as np
import pandas as pd
import os
import scipy.stats as stats
import sys, time
sys.path.append('../../joe/scripts/')
from train import *
from sklearn import cross_validation, svm
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from operator import itemgetter

parser = argparse.ArgumentParser(description='LOOCV over dataset')
parser.add_argument('--fm', type=str, help='feature matrix',
        default='../../data/features/cand_parse_all_2014_feat_matrix_unnormed_trimmed.pkl')
parser.add_argument('--cf', type=str, help='file of candidates and'
        ' CIDs', default='../../joe/out/cand_parse_all.dat')
parser.add_argument('--pltd', action='store_true', help='plot dwn' 
        ' distribution', default=False)
parser.add_argument('--ipnb', action='store_true', help='open ipython'
        ' notebook', default=False)
args, unknown = parser.parse_known_args()

def report(grid_scores, n_top=10):
        top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format( \
                score.mean_validation_score, np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters)) 
            print("")


def main():
    # read in data
    data = pd.read_pickle(args.fm)
    cols = ['name', 'state', 'CID', 'party', 'DWN-0', 'DWN-1']
    cands = pd.read_csv(args.cf, sep='\t', names=cols)
    cids = np.array(cands['CID'])
    dwn0 = np.array(cands['DWN-0'])
    dwn1 = np.array(cands['DWN-1'])

    if args.pltd:
        import seaborn as sns 
        import matplotlib.pyplot as plt
        cm = sns.diverging_palette(20, 220, n=2)
        sns.set(font_scale=.8)
        sns.set_style(style='white')
        party_df = sns.lmplot(x='DWN-0', y='DWN-1', hue='party', data=cands,
                fit_reg=False, palette=cm, legend=False)
        plt.legend(loc='upper right')
        plt.title('distribution of DW-NOMINATE scores by party')
        party_df.savefig('../../data/out/dwn_nooutlier.png', dpi=400, bbox_inches='tight')

    # training set and tragets 
    X_raw, DWN_0, DWN_1 = np.array(data), dwn0.T, dwn1.T
    
    # partition data into training and test set for each dwn score
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X_raw, DWN_0, test_size=0.2, random_state=0)
    X_train_2, X_test_2, Y_train_2, Y_test_2 = cross_validation.train_test_split(X_raw, DWN_1, test_size=0.2, random_state=0)

    # randomized parameter search dat shit
    clf = svm.SVR()

    # parameter distributions. initially we will just search over
    # different orders of magnitude of the parameters.
    param_grid = [{'kernel': ['rbf'], 
                    'gamma': [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                        'C': [1e-2, 1e-1, 1, 10, 100, 1000]
                   },

                  {'kernel': ['linear'],
                        'C': [1e-2, 1e-1, 1, 10, 100, 1000]
                   },

                  {'kernel': ['poly'],
                        'C': [1e-2, 1e-1, 1, 10, 100, 1000],
                   'degree': [2, 3, 4, 5]
                   }
                 ]

    # create the search
    grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=4)

    t0 = time.time()

    # for SVC, need to convert 
    # dw array to a boolean array for classification
    if isinstance(clf, svm.SVR):
        y_labels = []

        for i in DWN_0:
            if i > 0: y_labels.append(1)
            else: y_labels.append(0)
    
        y_labels = np.array(y_labels)

    # perform the search on development subset of data
    grid_search.fit(X_train, Y_train)
    print("RandomizedSearchCV took %.2f seconds." % (time.time() - t0))

    report(grid_search.grid_scores_)

    #clf_new = svm.SVR(kernel=params_2_use['kernel'], 
    #                  C=params_2_use['C'],
    #                  degree=params_2_use['degree']).fit(X_train, Y_train)
    #print(clf_new.score(X_test, Y_test))

    ## get minimum error results
    #lr = np.array(lr)
    #lr_min_err = lr[lr[:,1].argmin()]
    #svr = np.array(svr)
    #svr_min_err = svr[svr[:,1].argmin()]

    if args.ipnb:
        IPython.embed()

if __name__ == '__main__':
    main()
