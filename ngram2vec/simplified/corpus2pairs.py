from math import sqrt
from random import Random
from docopt import docopt
from corpus2vocab import getNgram
import sys
sys.path.append('./ngram2vec')
from representations.matrix_serializer import load_count_vocabulary
import six


def main():
    args = docopt("""
    Usage:
        corpus2pairs.py [options] <corpus> <vocab> <pairs>

    Options:
        --win NUM                  Window size [default: 2]
        --sub NUM                  Subsampling threshold [default: 0]
        --ngram_word NUM           (Center) word vocabulary includes grams of 1st to nth order [default: 1]
        --ngram_context NUM        Context vocabulary includes grams of 1st to nth order [default: 1]
        --overlap                  Whether overlaping pairs are allowed or not
    """)

    print ("**********************")
    print ("corpus2pairs")

    pairs_file = open(args['<pairs>'], 'w')
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
    print ('vocabulary size: ' + str(len(vocab)))
    with open(args['<corpus>']) as f:
        line_num = 0
        print (str(int(line_num/1000**1)) + "K lines processed.")
        for line in f:
            line_num += 1
            if ((line_num) % 1000) == 0:
                print ("\x1b[1A" + str(int(line_num/1000)) + "K lines processed.")
            line2features(line, args, vocab, pairs_file, sub, subsampler)

    pairs_file.close()
    print ("corpus2pairs finished")
                        

def check_word(t, vocab, sub, subsampler, rnd): #discard tokens
    if t is None:
        return None
    if sub:
        t = t if t not in subsampler or rnd.random() > subsampler[t] else None
        if t is None:
            return None
    t = t if t in vocab else None
    return t


def line2features(line, args, vocab, pairs_file, sub, subsampler):
    win = int(args['--win'])
    ngram_word = int(args['--ngram_word'])
    ngram_context = int(args['--ngram_context'])
    overlap = args['--overlap']
    rnd = Random(17)
    tokens = line.strip().split()
    for i in range(len(tokens)): #loop for each position in a line
        for gram_word in range(1, ngram_word+1): #loop for grams of different orders in (center) word 
            word = getNgram(tokens, i, gram_word)
            word = check_word(word, vocab, sub, subsampler, rnd)
            if word is None:
                continue
            for gram_context in range(1, ngram_context+1): #loop for grams of different orders in context
                start = i - win + gram_word - 1
                end = i + win - gram_context + 1
                for j in range(start, end + 1):
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
                    pairs_file.write(word + ' ' + context + "\n") #write pairs to the file



if __name__ == '__main__':
    main()
