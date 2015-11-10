#!/usr/bin/env python

import argparse
import IPython

parser = argparse.ArgumentParser(description='pluck specific congresses \
        from DW-nominate choice probability database')
parser.add_argument('--start', type=int, help='start session #', required=True)
parser.add_argument('--stop', type=int, help='end session #', required=True)
parser.add_argument('--fn_save', type=str, help='filename for saving')
# e.g. epsilon = 0.4 will pull all candidates whose ideal points are
# <-0.8, between -0.2 and 0.2, and >0.8
parser.add_argument('--epsilon', type=float, help='epsilon for defining width of \
        window about 0.0 and away from margins', required=True)
args = parser.parse_args()

# databases
FN_IDEAL_POINT_DATA = '../data/HL01113D21_BSSE.dat'
FN_STATE_CODES = '../data/STATE_CODES.dat'


def load_state_codes():
    '''
    handy database of the state codes used
    by the dw-nom guys
    '''
    state_codes = {}

    with open(FN_STATE_CODES, 'r') as f:
        for line in f:
            fields = line.split()
            state_codes[int(fields[0])] = fields[1]

    return state_codes


def parse_database(congress_start, congress_stop, epsilon):
    '''
    pluck congress members who fulfill provided
    ideal point criteria and dump to fn_save
    '''
    # it is annoying to have to parse this way but 
    # a couple of the columns are not delimited by whitespace
    # and the names seem inconsistent
    party_start = 25 
    party_end = party_start + 3
    state_code_start = 11
    state_code_end = state_code_start + 2
    nm_start = 29
    nm_end = 43
    state_codes = load_state_codes()
    saved = {}

    with open(FN_IDEAL_POINT_DATA, 'r') as f_dw:
        # line template for the numeric fields
        comment_line = "# [congress_number] [ideal_pt_dimension_1] [ideal_point_dimension_2] [state] [party] [name]\n"
        f_out_line_template = "{:03d}\t{: .4f}\t{: .4f}"
        f_out_moderate = open('../out/MODERATES.dat', 'w')
        f_out_margins  = open('../out/MARGINS.dat', 'w')
        f_out_moderate.write(comment_line)
        f_out_margins.write(comment_line)

        for line in f_dw:
            fields = line.split()
            congress_number = int(fields[0])

            if congress_number >= congress_start and congress_number <= congress_stop:
                ideal_pt_dim_1 = float(fields[-9])
                ideal_pt_dim_2 = float(fields[-8])

                if ideal_pt_dim_1 > -0.5 * epsilon and ideal_pt_dim_1 < 0.5 * epsilon:
                    type = 'MODERATE'
                elif ideal_pt_dim_1 > (1.0 - 0.5 * epsilon) or \
                     ideal_pt_dim_1 < (-1.0 + 0.5 * epsilon):
                    type = 'MARGINS'
                else:
                    continue
                try:
                    state = state_codes[int(line[state_code_start:state_code_end])]
                except:
                    state = 'NONE'

                party_code = int(line[party_start:party_end])

                if party_code == 100: party = 'D'
                elif party_code == 200: party = 'R'
                elif party_code == 328: party = 'I'
                # 'O' stands for 'other'. we really
                # should only have to deal with the three cases above
                else: party = 'O'

                name_segment = line[nm_start:nm_end].lower()
                segment_fields = name_segment.split()
                len_segment = len(segment_fields)

                if len_segment == 1:
                    name = segment_fields[0]
                else:
                    name = segment_fields[0] + ' ' + segment_fields[1][0]
                #else: raise RuntimeError('Name ' + name_segment + ' is cray.')

                # we will only pull one ideal point per candidate for
                # now, though it is possible that there are election
                # data for multiple campaigns
                if name in saved and saved[name] == state: continue
                else: saved[name] = state

                line_to_write = f_out_line_template.format(congress_number, 
                                                      ideal_pt_dim_1,
                                                      ideal_pt_dim_2) + \
                                                      '\t' + state + \
                                                      '\t' + party + \
                                                      '\t' + name + '\n'
                if type == 'MODERATE':
                    f_out_moderate.write(line_to_write)
                elif type == 'MARGINS':
                    f_out_margins.write(line_to_write)
                else: raise RuntimeError('You screwed up.')

        f_out_moderate.close()
        f_out_margins.close()
                                

def main():
    parse_database(args.start, args.stop, args.epsilon)


if __name__ == '__main__':
    main()
