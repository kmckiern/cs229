#!/bin/env python

import IPython
import argparse
import numpy as np
import pandas as pd
import os
import scipy.stats as stats
import sys, time, random
sys.path.append('../../joe/scripts/')
from train import *
from sklearn import cross_validation, svm
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from operator import itemgetter

parser = argparse.ArgumentParser(description='train the modelz')
parser.add_argument('--fm', type=str, help='feature matrix', default='../../data/features/cand_parse_all_fresh_2010_feat_matrix_trim_normed.pkl')
parser.add_argument('--cf', type=str, help='file of candidates and CIDs', default='../../joe/out/cand_parse_all.dat')
parser.add_argument('--pltd', action='store_true', help='plot dwn distribution', default=False)
parser.add_argument('--ipnb', action='store_true', help='open ipython notebook', default=False)
args, unknown = parser.parse_known_args()


def report(grid_scores, n_top=10):
    """
    borrowed from scikitlearn docs. reports top 10 classifiers
    found by the grid search.
    """
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format( \
                score.mean_validation_score, np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters)) 
        print("")


def main():
    # read in data
    data_2010 = np.array( pd.read_pickle(args.fm) )
    data_2012 = np.array( pd.read_pickle('../../data/features/cand_parse_all_fresh_2012_feat_matrix_trim_normed.pkl') )
    data_2014 = np.array( pd.read_pickle('../../data/features/cand_parse_all_fresh_2014_feat_matrix_trim_normed.pkl') )
    data_2016 = np.array( pd.read_pickle('../../data/features/cand_parse_all_fresh_2016_feat_matrix_trim_normed.pkl') )
    scores_2010 = np.array( pd.read_pickle('../../data/features/cand_parse_all_fresh_2010_scores.pkl') )
    scores_2012 = np.array( pd.read_pickle('../../data/features/cand_parse_all_fresh_2012_scores.pkl') )
    scores_2014 = np.array( pd.read_pickle('../../data/features/cand_parse_all_fresh_2014_scores.pkl') )
    scores_2016 = np.array( pd.read_pickle('../../data/features/cand_parse_all_fresh_2016_scores.pkl') )

    i, j = data_2010.shape[0], data_2010.shape[1]

    # randomly select ~7% of people and remove them
    # from each year. easiest to do this with masks
    mask = np.zeros(i,dtype=bool)
    # random.sample does sampling without replacement
    test_idx = np.array(random.sample(np.arange(i,dtype=int), int(0.067*i)))
    mask[test_idx] = True

    set_test = (np.concatenate((data_2010[mask], data_2012[mask], data_2014[mask]), axis=0),
                np.concatenate((scores_2010[:,0][mask], scores_2012[:,0][mask], scores_2014[:,0][mask]), axis=0),
                np.concatenate((scores_2010[:,1][mask], scores_2012[:,1][mask], scores_2014[:,1][mask]), axis=0)
               )

    set_train = (np.concatenate((data_2010[~mask], data_2012[~mask], data_2014[~mask]), axis=0),
                 np.concatenate((scores_2010[:,0][~mask], scores_2012[:,0][~mask], scores_2014[:,0][~mask]), axis=0),
                 np.concatenate((scores_2010[:,1][~mask], scores_2012[:,1][~mask], scores_2014[:,1][~mask]), axis=0)
                )

    X_train, X_test, Y_train, Y_test = set_train[0], set_test[0], set_train[1], set_test[1]

    # grid spacing
    Cs = np.arange(1,101,5)
    gammas = np.arange(10,210,10)

    param_grid = [{'kernel': ['rbf'], 
                   'gamma': gammas,
                       'C': Cs
                   }]

    grid_search = GridSearchCV(svm.SVR(), param_grid, cv=5, n_jobs=4)
    t0 = time.time()

    print('Performing search...')
    # perform the search on development subset of data
    grid_search.fit(X_train, Y_train)
    print('GridSearchCV took % .2f seconds.' % (time.time() - t0))
    print('')

    # use best performing training model to estimate the test set error
    clf_best = grid_search.best_estimator_

    # k-fold cross validation estimates the test score
    test_score = clf_best.score(X_test, Y_test)
    print('Test scores: %.4f for DWN0.' % (test_score))
    print('Done.')


    if args.ipnb:
        IPython.embed()


if __name__ == '__main__':
    main()
