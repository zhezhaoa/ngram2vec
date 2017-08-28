from docopt import docopt
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np
import random

from representations.matrix_serializer import save_matrix, save_vocabulary, load_vocabulary


def main():
    args = docopt("""
    Usage:
        counts2shuf.py [options] <counts> <output>
    
    Options:
        --memory_size NUM        Memory size available [default: 8.0]
    """)
    
    print "**********************"
    print "counts2shuf"

    counts = []
    with open(args['<counts>'], 'r') as f:
        counts_num = 0
        print str(counts_num/1000**2) + "M tokens processed."
        for line in f:
            print "\x1b[1A" + str(counts_num/1000**2) + "M tokens processed."
            counts_num += 1
            word, context, count = line.strip().split()
            counts.append((word, context, count))
    random.shuffle(counts)
    with open(args['<output>'], 'w') as f:
        for count in counts:
            f.write(count[0] + ' ' + count[1] + ' ' + count[2] + '\n')

if __name__ == '__main__':
    main()
