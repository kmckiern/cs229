#!/usr/bin/env python

import sys

fn_cand_parse = '../../data/candidates/cand_parse.dat'
fn_moderates  = '../out/ALL.dat'
fn_out = '../out/cand_parse_all.dat'

def main():
    f_out = open(fn_out, 'w')
    with open(fn_moderates, 'r') as f_mods:
        moderates = [line.split() for line in f_mods.readlines()[1:]] 
    with open(fn_cand_parse, 'r') as f:
        cand_parse_raw = [line.split() for line in f.readlines()[1:]]
    for line in cand_parse_raw:
        # exclude senate people and those with label 'pres'
        if len(line[1]) == 2:
            # check to see if they are in my db
            for db_line in moderates:
                if line[0].lower() in db_line and line[1] == db_line[-3]:
                    # bingo, we need to mark them
                    line_marked = '\t'.join(line) + '\t' + db_line[4] \
                                  + '\t' + '\t'.join(db_line[1:3]) + '\n'
                    f_out.write(line_marked)
    f_out.close()

if __name__ == '__main__':
    main()
