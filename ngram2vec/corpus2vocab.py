from docopt import docopt
from representations.matrix_serializer import save_count_vocabulary
from sys import getsizeof
import six


def main():
    args = docopt("""
    Usage:
        corpus2vocab.py [options] <corpus> <output>
    
    Options:
        --ngram NUM              Vocabulary includes grams of 1st to nth order [default: 1]
        --memory_size NUM        Memory size available [default: 8.0]
        --min_count NUM          Ignore words below a threshold [default: 10]
    """)

    print ("**********************")
    print ("corpus2vocab")
    ngram = int(args['--ngram'])
    memory_size = float(args['--memory_size']) * 1000**3
    min_count = int(args['--min_count'])
    vocab = {} # vocabulary (stored by dictionary)
    reduce_thr = 1 # remove low-frequency words when memory is insufficient
    memory_size_used = 0 # size of memory used by keys & values in dictionary (not include dictionary itself) 

    with open(args['<corpus>']) as f:
        tokens_num = 0
        print (str(int(tokens_num/1000**2)) + "M tokens processed.")
        for line in f:
            print ("\x1b[1A" + str(int(tokens_num/1000**2)) + "M tokens processed.") #ANSI
            tokens = line.strip().split()
            tokens_num += len(tokens)
            for pos in range(len(tokens)):            
                for gram in range(1, ngram+1):
                    token = getNgram(tokens, pos, gram)
                    if token is None :
                        continue
                    if token not in vocab :
                        memory_size_used += getsizeof(token)
                        vocab[token] = 1
                        if memory_size_used + getsizeof(vocab) > memory_size * 0.8: #reduce vocabulary when memory is insufficient
                            reduce_thr += 1
                            vocab_size = len(vocab)
                            vocab = {w: c for w, c in six.iteritems(vocab) if c >= reduce_thr}
                            memory_size_used *= float(len(vocab)) / vocab_size #estimate the size of memory used
                    else:
                        vocab[token] += 1

    vocab = {w: c for w, c in six.iteritems(vocab) if c >= min_count} #remove low-frequency words by pre-specified threshold, using six for bridging the gap between python 2 and 3
    vocab = sorted(six.iteritems(vocab), key=lambda item: item[1], reverse=True) #sort vocabulary by frequency in descending order
    save_count_vocabulary(args['<output>'], vocab)
    print ("number of tokens: " + str(tokens_num))
    print ("vocab size: " + str(len(vocab)))
    print ("low-frequency threshold: " + str(min_count if min_count > reduce_thr else reduce_thr))
    print ("corpus2vocab finished")


def getNgram(tokens, pos, gram): #uni:gram=1  bi:gram=2 tri:gram=3
    if pos < 0:
        return None
    if pos + gram > len(tokens):
        return None
    token = tokens[pos]
    for i in range(1, gram):
        token = token + "@$" + tokens[pos + i]
    return token


if __name__ == '__main__':
    main()
