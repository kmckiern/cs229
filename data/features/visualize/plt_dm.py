#!/bin/env python

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython

data_file = sys.argv[1]
data_name = data_file.split('.')[0]
df = pd.read_pickle(data_file)

# distill
data = np.array(df)
column_labels = df.columns
row_labels = df.index

# plot
fig, ax = plt.subplots()
fig.set_size_inches(16, 11)
heatmap = ax.pcolor(data, cmap='BuPu')
# aesthetix
ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
ax.set_yticklabels(row_labels)
ax.set_xticklabels(column_labels)
ax.invert_yaxis()
ax.xaxis.tick_top()
plt.xticks(rotation=90)
plt.tight_layout()

plt.xlim([0,data.shape[1]])
plt.ylim([0,data.shape[0]])

plt.savefig(sys.argv[2])
