from docopt import docopt
from scipy.stats.stats import spearmanr
import sys
sys.path.append('./ngram2vec/representations') 

from representations.representation_factory import create_representation


def main():
    args = docopt("""
    Usage:
        ws_eval.py [options] <representation> <representation_path> <task_path>
    
    Options:
        --neg NUM    Number of negative samples; subtracts its log from PMI (only applicable to PPMI) [default: 1]
        --w+c        Use ensemble of word and context vectors (not applicable to PPMI)
        --eig NUM    Weighted exponent of the eigenvalue matrix (only applicable to SVD) [default: 0.5]
    """)
    
    data = read_test_set(args['<task_path>'])
    representation = create_representation(args)
    correlation = evaluate(representation, data)
    print (args['<representation>'] + " " +  args['<task_path>'] + '\t%0.3f' % correlation)


def read_test_set(path):
    test = []
    with open(path) as f:
        for line in f:
            x, y, sim = line.strip().lower().split()
            test.append(((x, y), float(sim)))
    return test 


def evaluate(representation, data):
    results = []
    seen_num = 0
    for (x, y), sim in data:
        if representation.similarity(x, y) is not None :
            seen_num += 1
            results.append((representation.similarity(x, y), sim))
    actual, expected = zip(*results)
    print ("seen/total: " + str(seen_num) + "/" + str(len(data)))
    return spearmanr(actual, expected)[0]


if __name__ == '__main__':
    main()
