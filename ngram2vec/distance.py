from docopt import docopt
import numpy as np
from text2numpy import read_vectors


def main():
    args = docopt("""
    Usage:
        distance.py [options] <path>

    Options:
        --return_num NUM        Number of nearest neighbours returned [default: 50]
        --ngram NUM             Order of grams returened (0: return all grams) [default: 0]
    """)
    
    path = args['<path>']  
    return_num = int(args['--return_num'])
    ngram = int(args['--ngram'])
    join_str = "@$"
    vectors = read_vectors(path)
    index2word = vectors.keys()
    for w in index2word:
        vectors[w] = vectors[w]/np.sqrt((vectors[w]**2).sum())

    while(True):
        target = raw_input("Enter word or ngram (EXIT to break): ")
        target = join_str.join(target.strip().split())
        if target == "EXIT":
            break
    	if target not in index2word:
            print "Out of vocabulary word or ngram"
            continue
        target_vector = vectors[target]
        words_top = []
        for w in index2word:
            if ngram == 0:
                pass
            else:
                if ngram != w.count(join_str) + 1:
                    continue
            if target == w:
                continue
            sim = target_vector.dot(vectors[w])
            if len(words_top) == 0:
                words_top.append((w, sim))
                continue
            if sim <= words_top[len(words_top)-1][1] and len(words_top) >= return_num:
                continue
            for i in range(len(words_top)):
                if sim > words_top[i][1]:
                    break
            words_top.insert(i, (w, sim))
            if len(words_top) > return_num:
                words_top.pop(-1)
        print "words                cosine distance"
        for w, sim in words_top:
            print ("%-20s %-8.5f"%(w, sim))


if __name__ == '__main__':
    main()
