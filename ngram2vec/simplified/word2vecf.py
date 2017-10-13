# The code is based on the project on https://github.com/deborausujono/word2vecpy
# Modified by Zhe Zhao, Renmin university of China, Sem 2017
# Hierarchical-softmax and cbow are removed
# Supports arbitrary contexts 

from docopt import docopt
import sys
sys.path.append('./ngram2vec')
from representations.matrix_serializer import load_vocabulary, load_count_vocabulary
import numpy as np
import math

global_word_count = 0

class UnigramTable:
    def __init__(self, i2c, contexts):
        vocab_size = len(i2c)
        power = 0.75
        norm = sum([math.pow(contexts[i2c[i]], power) for i in range(len(i2c))])
        table_size = 1e8
        table = np.zeros(int(table_size), dtype=np.uint32)
        p = 0
        j = 0
        for i, c in enumerate(i2c):
            p += float(math.pow(contexts[c], power))/norm
            while j < table_size and float(j) / table_size < p:
                table[j] = i
                j += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]


def init_net(size, words_num, contexts_num):
    syn0 = np.random.uniform(low=-0.5/size, high=0.5/size, size=(words_num, size))
    syn1 = np.zeros(shape=(contexts_num, size))
    return (syn0, syn1)


def train_process(pairs_path, size, syn0, syn1, w2i, c2i, table, starting_alpha, negative, pairs_num, iters):
    global global_word_count
    lines_count = 0
    last_lines_processed = 0 
    lines_processed = 0
    alpha = starting_alpha
    pairs = []
    fi = open(pairs_path, 'r')
    for l in fi:
        lines_count += 1
        lines_processed += 1
        if len(l) == 0:
            continue
        pairs = l.split()
        pairs[0] = w2i[pairs[0]]
        pairs[1] = c2i[pairs[1]]

        if lines_processed % 10000 == 0:
            global_word_count += (lines_processed - last_lines_processed)
            last_lines_processed = lines_processed
            alpha = starting_alpha * (1 - float(global_word_count) / (pairs_num*iters))
            if alpha < starting_alpha * 0.0001: 
                alpha = starting_alpha * 0.0001
            sys.stdout.write("\r" + "Alpha: %f Progress: %d of %d (%.2f%%)" %(alpha, global_word_count, (pairs_num*iters), float(global_word_count) / (pairs_num*iters) * 100))

        neu1e = np.zeros(size)
        classifiers = [(pairs[1], 1)] + [(target, 0) for target in table.sample(negative)]
        for target, label in classifiers:
            z = np.dot(syn0[pairs[0]], syn1[target])
            p = sigmoid(z)
            g = alpha * (label - p)
            neu1e += g * syn1[target]
            syn1[target] += g * syn0[pairs[0]]
        syn0[pairs[0]] += neu1e

    global_word_count += (lines_processed - last_lines_processed)
    sys.stdout.write("\r" + "Alpha: %f Progress: %d of %d (%.2f%%)" %(alpha, global_word_count, (pairs_num*iters), float(global_word_count) / (pairs_num*iters) * 100))
    fi.close()


def save(i2w, syn0, fo):
    size = len(syn0[0])
    fo = open(fo, 'w')
    fo.write('%d %d\n' % (len(syn0), size))
    for word, vector in zip(i2w, syn0):
        vector_str = ' '.join([str(s) for s in vector])
        fo.write('%s %s\n' % (word, vector_str))
    fo.close()


def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))
    

def main():
    args = docopt("""
    Usage:
        word2vecf.py [options] <pairs> <words> <contexts> <outputs>

    Options:
        --negative NUM             Negative sampling [default: 5]
        --size NUM                 Embedding size [default: 100]
        --iters NUM                The number of iterations [default: 1]
    """)
    
    words_path = args['<words>']
    contexts_path = args['<contexts>']
    pairs_path = args['<pairs>']
    outputs_path = args['<outputs>']

    size = int(args['--size'])
    negative = int(args['--negative'])
    iters = int(args['--iters'])

    w2i, i2w = load_vocabulary(words_path)
    c2i, i2c = load_vocabulary(contexts_path)
    words = load_count_vocabulary(words_path)
    contexts = load_count_vocabulary(contexts_path)

    pairs_num = 0
    with open(pairs_path, 'r') as f:
        for l in f:
            pairs_num += 1

    alpha = 0.025
    syn0, syn1 = init_net(size, len(words), len(contexts))
    table = UnigramTable(i2c, contexts)
    for i in range(iters):
        train_process(pairs_path, size, syn0, syn1, w2i, c2i, table, alpha, negative, pairs_num, iters)
    save(i2w, syn0, outputs_path)
    print ("word2vecf finished")


if __name__ == '__main__':
    main()
