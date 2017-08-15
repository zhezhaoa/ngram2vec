from math import sqrt
from random import Random
import sys
from docopt import docopt
import multiprocessing
from corpus2vocab import getNgram
from representations.matrix_serializer import load_count_vocabulary



def main():
    args = docopt("""
    Usage:
        corpus2pairs.py [options] <corpus> <vocab> <pairs>

    Options:
        --thr NUM              The minimal word count for being in the vocabulary [default: 100]
        --win NUM              Window size [default: 2]
        --sub NUM              Subsampling threshold [default: 0]
        --ngram_word NUM       Ngram word [default: 1]
        --ngram_context NUM    Ngram context [default: 1]
        --threads NUM          The number of threads [default: 8]
        --overlap              Overlap
    """)
    print "corpus2pairs"
    threads = int(args['--threads'])
    threads_list = []
    for i in xrange(0, threads): 
        thread = multiprocessing.Process(target=c2p, args=(args, i))
        thread.start()
        threads_list.append(thread)
    for thread in threads_list:
        thread.join()
    print ""
    print "corpus2pairs finished"

def c2p(args, tid):
    pairs_file = open(args['<pairs>']+"_"+str(tid), 'w')
    thr = int(args['--thr'])
    win = int(args['--win'])
    subsample = float(args['--sub'])
    sub = subsample != 0
    ngram_word = int(args['--ngram_word'])
    ngram_context = int(args['--ngram_context'])
    overlap = args['--overlap']
    threads = int(args['--threads'])

    vocab = load_count_vocabulary(args['<vocab>'], thr)
    train_uni_num = 0
    for w, c in vocab.iteritems():
        if '@$' not in w:
            train_uni_num += c
    train_num = sum(vocab.values())
    if tid == 0:
        print 'vocabulary size: ' + str(len(vocab))
        print 'the number of training words (uni-grams): ' + str(train_uni_num)    
        print 'the number of training n-grams: ' + str(train_num)

    subsample *= train_uni_num
    if sub:
        subsampler = dict([(word, 1 - sqrt(subsample / count)) for word, count in vocab.items() if count > subsample])

    rnd = Random(17)
    with open(args['<corpus>']) as f:
        line_num = 0
        for line in f:
            line_num = line_num + 1
            if ((line_num) % 10000) == 0 and tid == threads - 1:
                sys.stdout.write("the number of (ten thousand) lines processed: " + str(line_num/10000) + "\r")
                sys.stdout.flush()
            if line_num % threads != tid:
                continue
            tokens = line.strip().split()
            for i in xrange(len(tokens)):
                for gram_word in xrange(1, ngram_word+1):
                    word = getNgram(tokens, i, gram_word)
                    word = check_word(word, vocab, sub, subsampler, rnd)
                    if word is None:
                        continue
                    for gram_context in xrange(1, ngram_context+1):
                        start = i - win + gram_word - 1
                        end = i + win - gram_context + 1
                        for j in xrange(start, end + 1):
                            if overlap:
                                if i == j and gram_word == gram_context:
                                    continue
                            else:
                                if len(set(range(i, i + gram_word)) & set(range(j, j + gram_context))) > 0:
                                    continue
                            context = getNgram(tokens, j, gram_context)
                            context = check_word(context, vocab, sub, subsampler, rnd)
                            if context is None:
                                continue
                            pairs_file.write(word + ' ' + context + "\n")
    pairs_file.close()
                        

def check_word(t, vocab, sub, subsampler, rnd):
    if t is None:
        return None
    if sub:
        t = t if t not in subsampler or rnd.random() > subsampler[t] else None
        if t is None:
            return None
    t = t if t in vocab else None
    return t


if __name__ == '__main__':
    main()

