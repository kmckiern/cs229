#!/bin/env python

import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser(description='LOOCV over dataset')
parser.add_argument('--fm', type=str, help='feature matrix', default='../../data/features/cand_parse_moderates_2014_feat_matrix.pkl')
parser.add_argument('--cf', type=str, help='file of candidates and CIDs', default='../../joe/out/test.dat')
parser.add_argument('--train', type=str, help='training method')
args, unknown = parser.parse_known_args()

def main():
    data = np.array(pd.read_pickle(args.fm))
    cols = ['name', 'state', 'CID', 'note', 'dwn0', 'dwn1']
    cands = pd.read_csv(args.cf, sep='\t', names=cols)
    scores = list(cands['dwn0'])

    method = args.train
    if method == 'LR':
        from whatevr import lr as train
    if method == 'SVR':
        from whatevr import svr as train

    errors = []
    models = []
    for ndx, d in enumerate(data):
        test = d
        train = np.delete(data, (ndx), axis=0)
        model, error = train.main(train, test)
        errors.append(error)
        models.append(model)

    avg_error = np.average(np.array(errors))
    best_model = models[errors.argmin()]

if __name__ == '__main__':
    main()
