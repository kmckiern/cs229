#!/bin/env python

"""
give a candidate, return vector of normalized funding values
"""

from crpapi import CRP, CRPApiError
import argparse
import json

parser = argparse.ArgumentParser(description='get voting and financial information for an input candidate')
parser.add_argument('--cid', type=str, help='CRP CID', default='N00007360')
parser.add_argument('--pref', type=str, help='output file directory preface', default='/Users/kerimckiernan/Documents/class/F15/working/cs229/data/out/')
parser.add_argument('--yr', type=str, help='query year', default='2012')
parser.add_argument('--ak', type=str, help='api key')
parser.add_argument('--write_dicts', action='store_true', help='write candidate data to dat file', default=False)
parser.add_argument('--ip', action='store_true', help='open ipython after variable declaration', default=False)
args = parser.parse_args()

def json_dict(d):
    return json.dumps(d, sort_keys=True, indent=2)

def dict_to_dat(fn, d):
    print >>open(fn,'w+'), json_dict(d)

def main():
    CRP.apikey = args.ak
    cid = args.cid
    yr = args.yr
    pref = args.pref
    
    #---------- CANDIDATE META INFO ----------#
    '''
    contribution information on a candidate for indicated cycle
    '''
    c_sum = CRP.candSummary.get(cid=cid, cycle=yr)
    
    #---------- FINANCE STREAMS ----------#
    '''
    member personal financial disclosure statement
    '''
    mem_pfd = CRP.memPFDprofile.get(cid=cid, year=yr)
    '''
    top contributors to a candidate/member for indicated period
    '''
    top_cont = CRP.candContrib.get(cid=cid, cycle=yr)
    '''
    top industries to a candidate/member for indicated period
    '''
    top_indust = CRP.candContrib.get(cid=cid, cycle=yr)
    '''
    top sectors to a candidate/member for indicated period
    '''
    top_sect = CRP.candSector.get(cid=cid, cycle=yr)

    if args.write_dicts:
        labels = {
            0   : 'c_sum',
            1   : 'mem_pfd',
            2   : 'top_cont',
            3   : 'top_indust',
            4   : 'top_sect',
        }
        
        # trying to decide features, going to write choices to file
        data_dicts = [c_sum, mem_pfd, top_cont, top_indust, top_sect]
        for ndx, i in enumerate(data_dicts):
            of = pref + labels[ndx] + '.dat'
            dict_to_dat(of, i)
    
    # if i want to interact with the data
    if args.ip:
        import IPython
        IPython.embed()

    '''
    not used: 

    '''

if __name__ == '__main__':
    main()
