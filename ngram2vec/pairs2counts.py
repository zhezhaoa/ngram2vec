from collections import Counter
from docopt import docopt
from math import sqrt
from sys import getsizeof
import numpy as np
from representations.matrix_serializer import load_vocabulary
import cPickle as pickle
import os


def main():
    # get all parameters.
    args = docopt("""
    Usage:
        pairs2counts.py [options] <pairs> <vocab_word> <vocab_context> <counts>

    Options:
        --memory_size NUM        Memory size available [default: 8.0]

    """)

    print "**********************"
    print "pairs2counts"

    wi, iw =load_vocabulary(args['<vocab_word>'])
    ci, ic = load_vocabulary(args['<vocab_context>'])
    max_product = 10000
    memory_size = float(args['--memory_size']) * 1000**3
    D = {} #store bottom-right part of co-occurrence matrix in dictionary
    tmpfile_num = 1
    memory_size_used = 0

    #store top-left corner of co-occurrence matrix in array, which is the strategy used in GloVe
    lookup = [0,]
    for i in xrange(len(iw)):
        if max_product / (i + 1) == 0:
            break
        if max_product / (i + 1) > len(iw):
            lookup.append(lookup[-1] + len(iw))
        else:
            lookup.append(lookup[-1] + max_product / (i + 1))
    M = np.zeros(lookup[-1] + 1, dtype=np.int32)

    with open(args['<pairs>']) as f:
        pairs_num = 0
        print str(pairs_num/1000**2) + "M pairs processed."
        for line in f:
            pairs_num += 1
            if pairs_num % 1000**2 == 0:
                print "\x1b[1A" + str(pairs_num/1000**2) + "M pairs processed."
            if getsizeof(D) + memory_size_used + getsizeof(M) > memory_size * 0.8: #write dictionary to disk when memory is insufficient
                with open(args['<counts>'] + '_' + str(tmpfile_num), 'wb') as f:
                    tmp_sorted = sorted(D.keys())
                    for i in tmp_sorted:
                        pickle.dump((i, D[i]), f, True)
                    D.clear()
                    memory_size_used = 0
                    tmpfile_num += 1
            pair = line.strip().split()
            word_index = wi[pair[0]]
            context_index = ci[pair[1]]
            if (word_index + 1) * (context_index + 1) <= max_product: #store top-left corner in M, which stays in memory all time
                M[lookup[word_index] + context_index] += 1  
            else: #store bottom-right part in D, which is written to disk when memory is insufficient
                if word_index in D:
                    tmp_size = getsizeof(D[word_index])
                    D[word_index].update({context_index: 1})
                    memory_size_used += getsizeof(D[word_index]) - tmp_size #estimate the size of memory used
                else:
                    D[word_index] = Counter({context_index: 1})
                    memory_size_used += getsizeof(D[word_index])
    with open(args['<counts>'] + '_' + str(tmpfile_num), 'wb') as f:
        tmp_sorted = sorted(D.keys())
        for i in tmp_sorted:
            pickle.dump((i, D[i]), f, True)
        D.clear()
        tmpfile_num += 1

    for i in xrange(len(lookup)): #transform M to dictionary structure
        D[i] = Counter()
        if i == len(lookup) - 1:
            if M[lookup[i] + j] > 0:
                D[i].update({j: M[lookup[i] + j]})
            break
        for j in xrange(lookup[i+1] - lookup[i]):
            if M[lookup[i] + j] > 0:
                D[i].update({j: M[lookup[i] + j]})
    with open(args['<counts>'] + '_' + str(0), 'wb') as f: #write top-left corner to disk
        tmp_sorted = sorted(D.keys())
        for i in tmp_sorted:
            pickle.dump((i, D[i]), f, True)
        D.clear()      


    #merge tmpfiles to co-occurrence matrix
    tmpfiles = []
    top_buffer = [] #store top elements of tmpfiles
    counts_num = 0
    counts_file = open(args['<counts>'], 'w')
    for i in xrange(tmpfile_num):
        tmpfiles.append(open(args['<counts>'] + '_' + str(i), 'rb'))
        top_buffer.append(pickle.load(tmpfiles[i]))
    old = top_buffer[0]
    top_buffer[0] = pickle.load(tmpfiles[0])
    print str(counts_num/1000**2) + "M counts processed."
    while True:
        arg_min = np.argmin(np.asarray([c[0] for c in top_buffer])) #find the element with smallest key (center word)
        if top_buffer[arg_min][0] == old[0]: #merge values when keys are the same
            old[1].update(top_buffer[arg_min][1])
        else:
            tmp_sorted = sorted(old[1].keys()) #write the old element when keys are different (which means all pairs whose center words are [old.key] are aggregated)
            for w in tmp_sorted:
                counts_num += 1
                if counts_num % 1000**2 == 0:
                    print "\x1b[1A" + str(counts_num/1000**2) + "M counts processed."
                counts_file.write(str(old[0]) + " " + str(w) + " " + str(old[1][w]) + "\n")
            old = top_buffer[arg_min]
        try:
            top_buffer[arg_min] = pickle.load(tmpfiles[arg_min])
        except EOFError: #when elements in file are exhausted
            top_buffer[arg_min] = (np.inf, Counter())
            tmpfile_num -= 1
        if tmpfile_num == 0:
            tmp_sorted = sorted(old[1].keys())
            for w in tmp_sorted:
                counts_num += 1
                counts_file.write(str(old[0]) + " " + str(w) + " " + str(old[1][w]) + "\n")
            break
    counts_file.close()
    print "number of counts: ", counts_num
    for i in xrange(len(top_buffer)): #remove tmpfiles
        os.remove(args['<counts>'] + '_' + str(i))

    print "pairs2counts finished"

if __name__ == '__main__':
    main()
