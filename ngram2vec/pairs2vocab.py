from docopt import docopt
from representations.matrix_serializer import save_count_vocabulary
import six

def main():
    args = docopt("""
    Usage:
        pairs2vocab.py <pairs> <words> <contexts>
    """)
    
    print ("**********************")
    print ("pairs2vocab")
    words_path = args['<words>']
    contexts_path = args['<contexts>']

    words = {} #center word vocabulary
    contexts = {} #context vocabulary
    with open(args['<pairs>']) as f:
        pairs_num = 0
        print (str(int(pairs_num/1000**2)) + "M pairs processed.")
        for line in f:
            pairs_num += 1
            if pairs_num % 1000**2 == 0:
                print ("\x1b[1A" + str(int(pairs_num/1000**2)) + "M pairs processed.")
            pair = line.strip().split()
            if pair[0] not in words :
                words[pair[0]] = 1
            else:
                words[pair[0]] += 1      
            if pair[1] not in contexts :
                contexts[pair[1]] = 1
            else:
                contexts[pair[1]] += 1

    words = sorted(six.iteritems(words), key=lambda item: item[1], reverse=True)
    contexts = sorted(six.iteritems(contexts), key=lambda item: item[1], reverse=True)

    save_count_vocabulary(words_path, words)
    save_count_vocabulary(contexts_path, contexts)   
    print ("words size: " + str(len(words)))
    print ("contexts size: " + str(len(contexts)))
    print ("number of pairs: " + str(pairs_num))
    print ("pairs2vocab finished")


if __name__ == '__main__':
    main()

