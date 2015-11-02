#!/bin/env python

"""
NOTES:
- many of these are accessible only for: 2012, 2014, 2016
- generalize how to get candidate data by parsing some of the lists in ../ref/
"""

from crpapi import CRP, CRPApiError
import argparse
import json
import IPython

parser = argparse.ArgumentParser(description='get voting and financial information for an input candidate')
parser.add_argument('--cid', type=str, help='CRP CID', default='N00007360')
parser.add_argument('--yr', type=str, help='query year', default='2012')
parser.add_argument('--ak', type=str, help='api key')
args = parser.parse_args()

def json_dict(d):
    return json.dumps(d, sort_keys=True, indent=2)

CRP.apikey = args.ak
cid = args.cid
yr = args.yr

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

IPython.embed()
