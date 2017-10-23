from docopt import docopt
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np
import random
from sys import getsizeof
import sys
import os

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

    #shuffle round 1
    memory_size = float(args['--memory_size']) * 1000**3
    counts = []
    counts_num_per_file = []
    tmp_id = 0
    with open(args['<counts>'], 'r') as f:
        counts_num = 0
        for line in f:
            if counts_num % 1000 == 0:
                sys.stdout.write("\r" + str(counts_num/1000**2) + "M counts processed.")
            counts_num += 1
            word, context, count = line.strip().split()
            counts.append((int(word), int(context), float(count)))
            if getsizeof(counts) + (getsizeof((int(0),int(0),float(0))) + getsizeof(int(0)) * 2 + getsizeof(float(0)) ) * len(counts) > memory_size:
                random.shuffle(counts)
                with open(args['<output>'] + str(tmp_id), 'w') as f:
                    for count in counts:
                        f.write(str(count[0]) + ' ' + str(count[1]) + ' ' + str(count[2]) + '\n')
                counts_num_per_file.append(counts_num)
                counts = []
                tmp_id += 1

    random.shuffle(counts)
    with open(args['<output>'] + str(tmp_id), 'w') as f:
        for count in counts:
            f.write(str(count[0]) + ' ' + str(count[1]) + ' ' + str(count[2]) + '\n')
        counts = []
        tmp_id += 1
    if tmp_id == 1:
        counts_num_per_file.append(counts_num)

    print "number of tmpfiles: ", tmp_id 

    #shuffle round 2
    counts_num = 0
    output_file = open(args['<output>'], 'w')
    tmpfiles = []
    for i in xrange(tmp_id):
        tmpfiles.append(open(args['<output>'] + str(i), 'r'))
    
    tmp_num = counts_num_per_file[0] / tmp_id
    for i in xrange(tmp_id - 1):
        counts = []
        for f in tmpfiles:
            for j in xrange(tmp_num):
                line = f.readline()
                if len(line) > 0:
                    if counts_num % 1000 == 0:
                        sys.stdout.write("\r" + str(counts_num/1000**2) + "M counts processed.")
                    counts_num += 1
                    word, context, count = line.strip().split()
                    counts.append((int(word), int(context), float(count)))
        random.shuffle(counts)
        for count in counts:
            output_file.write(str(count[0]) + ' ' + str(count[1]) + ' ' + str(count[2]) + '\n')
    counts = []
    for f in tmpfiles:
        for line in f:
            if counts_num % 1000 == 0:
                sys.stdout.write("\r" + str(counts_num/1000**2) + "M counts processed.")
            counts_num += 1
            word, context, count = line.strip().split()
            counts.append((int(word), int(context), float(count)))
    random.shuffle(counts)
    for count in counts:
        output_file.write(str(count[0]) + ' ' + str(count[1]) + ' ' + str(count[2]) + '\n')

    for i in xrange(tmp_id):
        tmpfiles[i].close()
    for i in xrange(tmp_id):
        os.remove(args['<output>'] + str(i))
    output_file.close()    
    print "number of counts: ", counts_num

if __name__ == '__main__':
    main()
