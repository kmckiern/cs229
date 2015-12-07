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
args = parser.parse_args()

sns.set(font_scale=1.0)
sns.set(font='Open Sans')
sns.set_style(style='white')
cm = 'coolwarm'

def main():
    opref = args.opref

    data_file = args.df
    data_name = data_file.split('/')[-1].split('.')[0]
    df = pd.read_pickle(data_file)
    n_cat = df.shape[-1]
    cats = df.columns

    svs = np.genfromtxt('../out_data/support_vectors.dat')
    lbls = np.zeros(df.shape[0])
    cat_numerical = np.arange(df.shape[1])
    for ndx, i in enumerate(svs):
        lbls[i] = ndx+1
    cp = sns.color_palette(cm, n_colors=len(lbls))
    df['SV'] = lbls

    xnp = np.array(df)
    for ndx, i in enumerate(xnp):
        if i[-1] == 0:
            plt.plot(i[:-1], cat_numerical, 's', color='.9', alpha=.1)
    for ndx, i in enumerate(xnp):
        if i[-1] != 0:
            plt.plot(i[:-1], cat_numerical, 'o', color=cp[ndx])

    sns.despine()
    plt.ylim([-1,12])
    plt.yticks(cat_numerical, cats, rotation='horizontal')
    plt.title('Support Vector Features')
    plt.xlabel('Normalized Contribution')
    plt.savefig(opref + data_name + '_SVs.png', dpi=400, bbox_inches='tight')

if __name__ == '__main__':
    main()    
