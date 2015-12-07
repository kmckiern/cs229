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

parser = argparse.ArgumentParser(description='train the modelz')
parser.add_argument('--fm', type=str, help='feature matrix', default='../../data/features/cand_parse_all_fresh_2010_feat_matrix_trim_normed_pc10.pkl')
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

    ## development - use data frames for uniform exclusion of a legislator across all years
    # read in data
    dfs =  glob.glob('../../data/features/*feat*norm*pkl')
    d2010, d2012, d2014 = [pd.read_pickle(i) for i in dfs]
    all_data = pd.read_pickle('../../data/features/cand_2010_2012_2014_fm_trim_normed.pkl')
    data_matrix = np.array(all_data)

    scores =  glob.glob('../../data/features/*feat*score*pkl')
    s2010, s2012, s2014 = [pd.read_pickle(i) for i in scores]
    all_scores = pd.read_pickle('../../data/features/cand_2010_2012_2014_scores.pkl')
    score_matrix = np.array(all_scores)

    fin_and_dwn = all_data + all_scores
    # detemine set of candidates in data frames
    cands = set(set3_df.index)
    ##

    data_2010 = np.array( pd.read_pickle(args.fm) )
    data_2012 = np.array( pd.read_pickle('../../data/features/cand_parse_all_fresh_2012_feat_matrix_trim_normed_pc10.pkl') )
    data_2014 = np.array( pd.read_pickle('../../data/features/cand_parse_all_fresh_2014_feat_matrix_trim_normed_pc10.pkl') )
    data_2016 = np.array( pd.read_pickle('../../data/features/cand_parse_all_fresh_2016_feat_matrix_trim_normed_pc10.pkl') )
    scores_2010 = np.array( pd.read_pickle('../../data/features/cand_parse_all_fresh_2010_scores.pkl') )
    scores_2012 = np.array( pd.read_pickle('../../data/features/cand_parse_all_fresh_2012_scores.pkl') )
    scores_2014 = np.array( pd.read_pickle('../../data/features/cand_parse_all_fresh_2014_scores.pkl') )
    scores_2016 = np.array( pd.read_pickle('../../data/features/cand_parse_all_fresh_2016_scores.pkl') )

    set1 = (data_2010, scores_2010[:,0], scores_2010[:,1])
    set2 = (np.concatenate((data_2010, data_2012), axis=0),
            np.concatenate((scores_2010[:,0], scores_2012[:,0]), axis=0),
            np.concatenate((scores_2010[:,1], scores_2012[:,1]), axis=0)
           )
    set3 = (np.concatenate((data_2010, data_2012, data_2014), axis=0),
            np.concatenate((scores_2010[:,0], scores_2012[:,0], scores_2014[:,0]), axis=0),
            np.concatenate((scores_2010[:,1], scores_2012[:,1], scores_2014[:,1]), axis=0)
           )
    set4 = (np.concatenate((data_2010, data_2012, data_2014, data_2016), axis=0),
            np.concatenate((scores_2010[:,0], scores_2012[:,0], scores_2014[:,0], scores_2016[:,0]), axis=0),
            np.concatenate((scores_2010[:,1], scores_2012[:,1], scores_2014[:,1], scores_2016[:,1]), axis=0)
           )

    for set in [set3]:

        # training set and tragets 
        X_raw, DWN_0, DWN_1 = set[0], set[1].T, set[2].T
        percent_train = np.arange(0.1,1.0,0.1)

        # partition data into training and test set for each dwn score
        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X_raw, DWN_0, test_size=0.2, random_state=0)
        X_train_2, X_test_2, Y_train_2, Y_test_2 = cross_validation.train_test_split(X_raw, DWN_1, test_size=0.2, random_state=0)

        print('Training set size: %d samples' % len(X_train))
        print('Test set size: %d samples' % len(X_test))
        print('')

        # define grid spacing here
        Cs = np.arange(1,101,1)
        gammas = np.arange(10,210,10)

        # parameter distributions. initially we will just search over
        # different orders of magnitude of the parameters.
        param_grid = [#{'kernel': ['rbf'], 
                      #  'gamma': gammas,
                      #      'C': Cs
                      # },

                       #{'kernel': ['linear'],
                       #     'C': Cs
                       #}#,

                      {'kernel': ['poly'],
                            'C': Cs,
                        'gamma': gammas,  
                        'degree': [2, 3, 4]
                       }
                     ]

        # create the search, one for each DWN dimension.
        grid_search_0 = GridSearchCV(svm.SVR(), param_grid, cv=5, n_jobs=4)
        #grid_search_1 = GridSearchCV(svm.SVR(), param_grid, cv=5, n_jobs=4)
        #grid_search_SVC = GridSearchCV(svm.SVC(), param_grid, cv=5, n_jobs=4)
        t0 = time.time()

        # for SVC, need to convert 
        # dw array to a boolean array for classification
        Y_train_SVC, Y_test_SVC = [], []
        
        for i in range(len(Y_train)):
            if Y_train[i] > 0: Y_train_SVC.append(1)
            else: Y_train_SVC.append(0)

        for i in range(len(Y_test)):   
            if Y_test[i] > 0: Y_test_SVC.append(1)
            else: Y_test_SVC.append(0)
        
        Y_train_SCV, Y_test_SVC = np.array(Y_train_SVC), np.array(Y_test_SVC)

        print('Performing search...')
        # perform the search on development subset of data
        grid_search_0.fit(X_train, Y_train)
        #grid_search_1.fit(X_train_2, Y_train_2)
        #grid_search_SVC.fit(X_train, Y_train_SVC)
        print('GridSearchCV took % .2f seconds.' % (time.time() - t0))
        print('')

        print('Grid search over DWN0:')
        report(grid_search_0.grid_scores_)
        print('')

        print('Grid search for DWN1:')
        #report(grid_search_1.grid_scores_)
        print('')

        print('Grid search for SVC:')
        #report(grid_search_SVC.grid_scores_)
        print('')

        # use best performing training model to estimate the test set error
        clf_best_0 = grid_search_0.best_estimator_
        #clf_best_1 = grid_search_1.best_estimator_
        #clf_best_SVC = grid_search_SVC.best_estimator_

        # k-fold cross validation estimates the test score
        test_score_0 = clf_best_0.score(X_test, Y_test) #np.average( cross_validation.cross_val_score(clf_best_0, X_test, y=Y_test, cv=2) )
        test_score_1 = 0.0 #clf_best_1.score(X_test, Y_test_2) #np.average( cross_validation.cross_val_score(clf_best_1, X_test_2, y=Y_test_2, cv=2) )
        test_score_SVC = 0.0 #clf_best_SVC.score(X_test, Y_test_SVC) #np.average( cross_validation.cross_val_score(clf_best_SVC, X_test, y=Y_test_SVC, cv=2) )
        print('Test scores: (%.4f, %.4f, %.4f) for DWN0, DWN1, and SVC, respectively.' % (test_score_0, test_score_1, test_score_SVC))
        print('Done.')


    if args.ipnb:
        IPython.embed()


if __name__ == '__main__':
    main()
