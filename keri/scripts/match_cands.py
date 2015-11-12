#!/bin/env python

"""
Return candidate information across: 
    cols = ['CID', 'CRPName', 'Party', 'DistIDRunFor', 'FECCandID']
"""

import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='get voting and financial information for an input candidate')
parser.add_argument('--id_file', type=str, help='crp ids xls file', default='../ref/CRP_IDs.xls')
parser.add_argument('--field', type=str, help='query by which field', default='CRPName')
parser.add_argument('--val', type=str, help='field value (eg. \'Ryan, Paul\'', default='Ryan, Paul')
parser.add_argument('--ip', action='store_true', help='open ipython after variable declaration', default=False)
args = parser.parse_args()

def main():
    cols = ['CID', 'CRPName', 'Party', 'DistIDRunFor', 'FECCandID']
    ids = pd.read_excel(args.id_file, columns=cols, skiprows=13)

    data = ids[ids[args.field] == args.val]
    print data
    
    # if i want to interact with the data
    if args.ip:
        import IPython
        IPython.embed()

if __name__ == '__main__':
    main()
