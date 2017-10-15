from math import sqrt
from random import Random
from docopt import docopt
import multiprocessing
from corpus2vocab import getNgram
from representations.matrix_serializer import load_count_vocabulary
import six
import sys
from line2features import ngram_ngram, word_word, word_text, word_wordLR, word_wordPos


def main():
    args = docopt("""
    Usage:
        corpus2pairs.py [options] <corpus> <vocab> <pairs>

    Options:
        --win NUM                  Window size [default: 2]
        --sub NUM                  Subsampling threshold [default: 0]
        --ngram_word NUM           (Center) word vocabulary includes grams of 1st to nth order [default: 1]
        --ngram_context NUM        Context vocabulary includes grams of 1st to nth order [default: 1]
        --threads_num NUM          The number of threads [default: 8]
        --overlap                  Whether overlaping pairs are allowed or not
    """)

    print ("**********************")
    print ("corpus2pairs")
    threads_num = int(args['--threads_num'])
    threads_list = []
    for i in range(0, threads_num): #extract pairs from corpus through multipule threads
        thread = multiprocessing.Process(target=c2p, args=(args, i))
        thread.start()
        threads_list.append(thread)
    for thread in threads_list:
        thread.join()
    print ("corpus2pairs finished")


def c2p(args, tid):
    pairs_file = open(args['<pairs>']+"_"+str(tid), 'w')
    threads_num = int(args['--threads_num'])
    subsample = float(args['--sub'])
    sub = subsample != 0
    vocab = load_count_vocabulary(args['<vocab>']) #load vocabulary (generated in corpus2vocab stage)
    train_uni_num = 0 #number of (unigram) tokens in corpus
    for w, c in six.iteritems(vocab):
        if '@$' not in w:
            train_uni_num += c
    train_num = sum(vocab.values()) #number of (ngram) tokens in corpus
    subsample *= train_uni_num
    if sub:
        subsampler = dict([(word, 1 - sqrt(subsample / count)) for word, count in six.iteritems(vocab) if count > subsample]) #subsampling technique
    if tid == 0:
        print ('vocabulary size: ' + str(len(vocab)))
    with open(args['<corpus>']) as f:
        line_num = 0
        for line in f:
            line_num += 1
            if ((line_num) % 1000) == 0 and tid == 0:
                sys.stdout.write("\r" + str(int(line_num/1000)) + "K lines processed.")
                sys.stdout.flush()
            if line_num % threads_num != tid:
                continue
            ngram_ngram(line, args, vocab, pairs_file, sub, subsampler)
            # word_word(line, args, vocab, pairs_file, sub, subsampler)
            # word_text(line, args, vocab, pairs_file, sub, subsampler, line_num)
            # word_wordPos(line, args, vocab, pairs_file, sub, subsampler)

    pairs_file.close()


if __name__ == '__main__':
    main()
