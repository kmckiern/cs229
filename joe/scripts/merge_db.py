#!/usr/bin/env python

import argparse, os, sys


parser = argparse.ArgumentParser(description='pluck candidates from my
        databse for whom we also have financial data')
parser.add_argument('--fn_cand_parse', type=str, help="keri's cand parse file name", required=True)
parser.add_argument('--fn_dw_data', type=str, help='my dwn database filename', default='../out/ALL.dat')
parser.add_argument('--fn_save', type=str, help='save to this file', required=True)
args = parser.parse_args()


def main():
    f_out = open(args.fn_save, 'w')

    with open(args.fn_dw_data, 'r') as f_dwn:
        peeps = [line.split() for line in f_dwn.readlines()[1:]]
    
    # since we're going to save only most recent score, 
    # reverse the database and parse it in that order
    peeps = list( reversed(peeps) )

    with open(args.fn_cand_parse, 'r') as f_cand_parse:
        cand_parse_raw = [line.split() for line in f_cand_parse.readlines()[1:]]

    for line in cand_parse_raw:
        # exclude senate people and those with label 'pres'
        if len(line[1]) == 2:
            # check to see if they are in my db
            for db_line in peeps:
                if line[0].lower() in db_line and \
                   line[1] == db_line[-3] and \
                   # TO DO: add condition for checking party affiliation
                   True:
                    # bingo, we need to mark them
                    line_marked = '\t'.join(line) + '\t' + db_line[4] \
                                  + '\t' + '\t'.join(db_line[1:3]) + '\n'
                    f_out.write(line_marked)
    f_out.close()


if __name__ == '__main__':
    main()
