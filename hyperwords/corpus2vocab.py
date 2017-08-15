from docopt import docopt
from representations.matrix_serializer import save_count_vocabulary
import sys


def main():
    args = docopt("""
    Usage:
        corpus2vocab.py [options] <corpus> <output>
    
    Options:
        --ngram NUM              ngram [default: 1]
        --reduce_thr NUM         reduce vocabulary when a certain number of lines are processed [default: 0]
    """)

    print "corpus2vocab"
    ngram = int(args['--ngram'])
    reduce_thr = int(args['--reduce_thr'])
    vocab = {}

    with open(args['<corpus>']) as f:
        line_num = 0
        for line in f:
            line_num = line_num + 1
            if line_num % 10000 == 0:
                sys.stdout.write("the number of (ten thousand) lines processed: " + str(line_num/10000) + "\r")
                sys.stdout.flush()
            if reduce_thr == 0:
                pass
            else:
                if line_num % reduce_thr == 0:
                    reduceVocab(vocab)
            tokens = line.strip().split()
            for pos in xrange(len(tokens)):            
                for gram in xrange(1, ngram+1):
                    token = getNgram(tokens, pos, gram)
                    if token is not None :
                        if token not in vocab :
                            vocab[token] = 1
                        else:
                            vocab[token] = vocab[token] + 1
    print ""
    print "the number of lines: " + str(line_num)
    print "vocab size: " + str(len(vocab))
    
    save_count_vocabulary(args['<output>'], vocab)
    print "corpus2vocab finished"


def getNgram(tokens, pos, gram): #uni:gram=1  bi:gram=2 tri:gram=3
    if pos < 0:
        return None
    if pos + gram > len(tokens):
        return None
    token = tokens[pos]
    for i in xrange(1, gram):
        token = token + "@$" + tokens[pos + i]
    return token

def reduceVocab(vocab):
    for w, c in vocab.items():
        if int(c) <= 1:
            vocab.pop(w)


if __name__ == '__main__':
    main()
