#!/bin/env python

import matplotlib
matplotlib.use('Agg')

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import IPython

parser = argparse.ArgumentParser(description='plot pd dataframe as heatmap')
parser.add_argument('--df', type=str, help='pd df pickle')
parser.add_argument('--opref', type=str, help='name of output file directory',
        default='../../data/features/visualize/')
parser.add_argument('--pca', type=int, help='number of PCA'
        'components', default=0)
args = parser.parse_args()

sns.set(font_scale=1.2)
sns.set(font='Open Sans')
sns.set_style(style='white')
cm = 'coolwarm'

def main():
    opref = args.opref

    data_file = args.df
    data_name = data_file.split('/')[-1].split('.')[0]
    df = pd.read_pickle(data_file)
    df_col = df.columns

    cmin = 0 # float(df.min().min())
    cmax = float(df.max().max())

    grid_kws = {"height_ratios": (.95, .05), "hspace": .25}
    fig, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    ax = sns.heatmap(df.T, vmin=cmin, vmax=cmax, cmap=cm, ax=ax, 
            xticklabels=False, cbar_ax=cbar_ax,
            cbar_kws={'orientation': 'horizontal'})
    ax.set_title('Normalized Campaign Contributions by Sector')
    ax.set_xlabel('Candidate')
    plt.yticks(rotation=0)
    fig.savefig(opref + data_name + '_feature_hm.png', bbox_inches='tight')

    nc = args.pca
    if nc > 0:
        from sklearn.decomposition import PCA
        pca = PCA(n_components = nc)
        pca.fit(df)
        PCs = pca.components_
        cols = ['PC ' + str(i) for i in range(PCs.shape[0])]
        pc_df = pd.DataFrame(PCs.T, columns=cols)
        pc_df.index = df_col

        grid_kws = {"height_ratios": (.95, .05), "hspace": .25}
        fig, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
        ax = sns.heatmap(pc_df, cmap=cm, ax=ax, 
                xticklabels=False, cbar_ax=cbar_ax,
                cbar_kws={'orientation': 'horizontal'})
        ax.set_title('Feature Matric Top Principal Components')
        plt.yticks(rotation=0)
        ax.set_xlabel('PC')
        fig.savefig(opref + data_name + '_pc_hm.png', bbox_inches='tight')

        plt.clf()
        g = sns.PairGrid(pc_df)
        g.map_lower(plt.hexbin, gridsize=30, mincnt=1, cmap=cm,
                edgecolor='none')
        g.map_diag(sns.kdeplot, lw=1, legend=False)
        g.savefig(opref + data_name + '_pc_pg.png', bbox_inches='tight')

if __name__ == '__main__':
    main()    
