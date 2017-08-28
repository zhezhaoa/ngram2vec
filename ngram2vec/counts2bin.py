from docopt import docopt
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np
import random
import struct


def main():
    args = docopt("""
    Usage:
        counts2bin.py <counts> <output>

    """)
    
    print "**********************"
    print "counts2bin"

    bin_file = open(args['<output>'], 'wb')
    with open(args['<counts>'], 'r') as f:
        counts_num = 0
        print str(counts_num/1000**2) + "M tokens processed."
        for line in f:
            print "\x1b[1A" + str(counts_num/1000**2) + "M tokens processed."
            counts_num += 1
            word, context, count = line.strip().split()
            b = struct.pack('iid', int(word), int(context), float(count))
            bin_file.write(b)
    print "number of counts: " + str(counts_num)
    bin_file.close()

if __name__ == '__main__':
    main()
