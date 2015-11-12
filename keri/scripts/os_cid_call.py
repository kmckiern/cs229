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

parser = argparse.ArgumentParser(description='get financial information for an input candidate')
parser.add_argument('--cid', type=str, help='CRP CID', default='N00007360')
parser.add_argument('--yr', type=str, help='query year', default='2012')
parser.add_argument('--ak', type=str, help='api key')
parser.add_argument('--categories', type=str, help='sector code database file', default='../../data/candidates/CRP_Categories.txt')
parser.add_argument('--categories', type=str, help='sector code database file', default='../ref/CRP_Categories.txt')
parser.add_argument('--write_dicts', action='store_true', help='write candidate data to dat file', default=False)
parser.add_argument('--pref', type=str, help='output file directory preface', default='../../data/out/examples/npelosi/')
parser.add_argument('--ip', action='store_true', help='open ipython after variable declaration', default=False)
args = parser.parse_args()

def json_dict(d):
    return json.dumps(d, sort_keys=True, indent=2)

def dict_to_dat(fn, d):
    print >>open(fn,'w+'), json_dict(d)

def d_norm(categories, data):
    num = len(categories)

    fin_data = np.zeros((2, num))
    for d in data:
        d_key = d['@attributes']['sector_name']
        col = categories.index(d_key)
        fin_data[0, col] = float(d['@attributes']['indivs'])
        fin_data[1, col] = float(d['@attributes']['pacs'])

    row_sums = fin_data.sum(axis=1)
    fin_data = fin_data / row_sums[:, np.newaxis]

    return fin_data

def main():
    CRP.apikey = args.ak
    cid = args.cid
    yr = args.yr
    pref = args.pref

    #---------- CANDIDATE FINANCE DATA ----------#
    '''
    top sectors to a candidate/member for indicated period
    '''
    top_sect = CRP.candSector.get(cid=cid, cycle=yr)

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
    
    #---------- GET FINANCE VECTOR ----------#
    categories = pd.read_csv(args.categories, sep='\t', skiprows=8)
    '''
    top sector normalized
    '''
    cats = list(set(categories['Sector']))
    sect_fin_data = d_norm(cats, top_sect)

    #---------- interact with data? ----------#
    if args.ip:
        import IPython
        IPython.embed()

if __name__ == '__main__':
    main()
