#!/bin/env python

"""
example usage (to write data to file, and open ipython):
    >> python py_call.py --ak (key) --write --ip

for a given candidate, get vector of financial information

NOTES:
- many of these are accessible only for: 2012, 2014, 2016
- generalize how to get candidate data by parsing some of the lists in ../ref/
"""

from crpapi import CRP, CRPApiError
import argparse
import json
import pandas as pd
import numpy as np
import sys

parser = argparse.ArgumentParser(description='get financial information for an input candidate')
parser.add_argument('--cid', type=str, help='CRP CID', default='N00007360')
parser.add_argument('--yr', type=str, help='query year', default='2014')
parser.add_argument('--ak', type=str, help='api key')
parser.add_argument('--categories', type=str, help='sector code database file', default='../../data/candidates/CRP_Categories.txt')
parser.add_argument('--write_dicts', action='store_true', help='write candidate data to dat file', default=False)
parser.add_argument('--pref', type=str, help='output file directory preface', default='../../data/out/examples/npelosi/')
parser.add_argument('--norm', action='store_true', help='normalize fin data', default=False)
parser.add_argument('--ip', action='store_true', help='open ipython after variable declaration', default=False)
args, unknown = parser.parse_known_args()

def json_dict(d):
    return json.dumps(d, sort_keys=True, indent=2)

def dict_to_dat(fn, d):
    print >>open(fn,'w+'), json_dict(d)

def group_data(categories, data, norm=False):
    num = len(categories)
    fin_data = np.zeros(num)
    features = []
    for d in data:
        d_key = d['@attributes']['sector_name']
        if d_key == 'Other':
            continue
        col = categories.index(d_key)
        fin_data[col] = float(d['@attributes']['pacs']) + float(d['@attributes']['indivs'])
    if norm:
        row_sums = fin_data.sum()
        fin_data = fin_data / row_sums
    return fin_data

def main(args):
    CRP.apikey = args.ak
    cid = args.cid
    yr = args.yr
    pref = args.pref

    #---------- GET FINANCE VECTOR ----------#
    categories = pd.read_csv(args.categories, sep='\t', skiprows=8)
    '''
    top sector normalized
    '''
    cats = list(set(categories['Sector']))
    cats.remove('Other')

    #---------- CANDIDATE FINANCE DATA ----------#
    '''
    top sectors to a candidate/member for indicated period
    '''
    try:
        top_sect = CRP.candSector.get(cid=cid, year=yr)
    except:
        print cid
        sys.exit()
    sect_fin_data = group_data(cats, top_sect, args.norm)

    #---------- write data? ----------#
    if args.write_dicts:
        #---------- CANDIDATE META INFO ----------#
        '''
        contribution information on a candidate for indicated cycle
        '''
        c_sum = CRP.candSummary.get(cid=cid, cycle=yr)
        labels = {
            0   : 'c_sum',
            1   : 'mem_pfd',
            2   : 'top_cont',
            3   : 'top_indust',
            4   : 'top_sect',
        }
        # trying to decide features, going to write choices to file
        data_dicts = [c_sum, top_sect]
        # data_dicts.append([mem_pfd, top_sect]
        for ndx, i in enumerate(data_dicts):
            of = pref + labels[ndx] + '.dat'
            dict_to_dat(of, i)
    
    #---------- interact with data? ----------#
    if args.ip:
        import IPython
        IPython.embed()

    return np.hstack(sect_fin_data), cats

if __name__ == '__main__':
    import sys
    main(args)
