import pandas as pd
import re

x = pd.read_excel('../../data/candidates/CRP_IDs.xls',header=12)
cids = list(x['CID'])
names = list(x['CRPName'])
parties = list(x['Party'])
dists = list(x['DistIDRunFor'])

# parse candidate names into first and last name
firsts = []
lasts = []
for i in names:
    l, f = i.split(',')
    f_only = f.split()[0]
    firsts.append(f_only)
    lasts.append(l)

# parse dist ID to get candidate state
states = []
for i in dists:
    try:
        states.append(str(re.match(r"(\w+)(\d+)", i, re.I).groups()[0][:2]))
    except:
        states.append(None)

all_data = zip(firsts, lasts, states, parties, cids)
of = open('../../data/candidates/cand_parse.dat', 'a+')
for i in all_data:
    line = '\t'.join(str(j) for j in i)
    of.write(line + '\n')
