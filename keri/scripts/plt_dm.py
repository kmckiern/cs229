#!/bin/env python

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import IPython

parser = argparse.ArgumentParser(description='plot pd dataframe as heatmap')
parser.add_argument('--df', type=str, help='pd df pickle')
parser.add_argument('--cols', type=str, help='column range')
parser.add_argument('--o', type=str, help='name of output file',
        default='features.png')
parser.add_argument('--pca', type=int, help='number of PCA'
        'components', default=0)
parser.add_argument('--pchm', type=str, help='name of PCA HM output file')
parser.add_argument('--pcpg', type=str, help='name of PCA PG output file')
args = parser.parse_args()

sns.set(font_scale=.8)
sns.set_style(style='white')
cm = 'YlGnBu_r'

def main():
    data_file = args.df
    data_name = data_file.split('.')[0]
    df = pd.read_pickle(data_file)
    df = df.drop('Other_pac', 1)
    df = df.drop('Other_indiv', 1)
    df_col = df.columns

    ax = sns.heatmap(df, cmap=cm)
    fig = plt.gcf()
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    fig.savefig(args.o)

    nc = args.pca
    if nc > 0:
        from sklearn.decomposition import PCA
        pca = PCA(n_components = nc)
        pca.fit(df)
        PCs = pca.components_
        cols = ['PC ' + str(i) for i in range(PCs.shape[0])]
        pc_df = pd.DataFrame(PCs.T, columns=cols)
        pc_df.index = df_col

        plt.clf()
        pc_hm = sns.heatmap(pc_df, cmap=cm)
        fig = plt.gcf()
        plt.yticks(rotation=0)
        fig.savefig(args.pchm)

        plt.clf()
        g = sns.PairGrid(pc_df)
        g.map_lower(plt.hexbin, gridsize=30, mincnt=1, cmap=cm,
                edgecolor='none')
        g.map_diag(sns.kdeplot, lw=1, legend=False)
        g.savefig(args.pcpg)

if __name__ == '__main__':
    main()    
