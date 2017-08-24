from docopt import docopt
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np

from representations.matrix_serializer import save_matrix, save_vocabulary, load_vocabulary


def main():
    args = docopt("""
    Usage:
        counts2pmi.py [options] <words_vocab> <contexts_vocab> <counts> <output>
    
    Options:
        --cds NUM    Context distribution smoothing [default: 1.0]
    """)
    
    counts_path = args['<counts>']
    vectors_path = args['<output>']
    words_path = args['<words_vocab>']
    contexts_path = args['<contexts_vocab>']
    cds = float(args['--cds'])

    counts = read_counts_matrix(words_path, contexts_path, counts_path)
    pmi = calc_pmi(counts, cds)
    save_matrix(vectors_path, pmi)


def read_counts_matrix(words_path, contexts_path, counts_path):
    """
    Reads the counts into a sparse matrix (CSR) from the count-word-context textual format.
    """

    w2i, i2w = load_vocabulary(words_path)
    c2i, i2c = load_vocabulary(contexts_path)
    counts = csr_matrix((len(w2i), len(c2i)), dtype=np.float32)
    tmp_counts = csr_matrix((len(w2i), len(c2i)), dtype=np.float32)
    update_threshold = 10 * 1000**2
    counts_num = 0
    i = 0
    row = []
    col = []
    data = []
    with open(counts_path) as f:
        print str(counts_num/1000**2) + "M counts processed."
        for line in f:
            if counts_num % 1000**2 == 0:
                print "\x1b[1A" + str(counts_num/1000**2) + "M counts processed."
            word, context, count = line.strip().split()
            row.append(int(word))
            col.append(int(context))
            data.append(int(float(count)))
            i += 1
            if i == update_threshold:
                i = 0
                row = np.asarray(row)
                col = np.asarray(col)
                data = np.asarray(data)
                tmp_counts = csr_matrix((data, (row, col)), shape=(len(w2i), len(c2i)), dtype=np.float32)
                counts = counts + tmp_counts
                row = []
                col = []
                data = []
            counts_num += 1
    row = np.asarray(row)
    col = np.asarray(col)
    data = np.asarray(data)
    tmp_counts = csr_matrix((data, (row, col)), shape=(len(w2i), len(c2i)), dtype=np.float32)
    counts = counts + tmp_counts

    return counts


def calc_pmi(counts, cds):
    """
    Calculates e^PMI; PMI without the log().
    """
    sum_w = np.array(counts.sum(axis=1))[:, 0]
    sum_c = np.array(counts.sum(axis=0))[0, :]
    if cds != 1:
        sum_c = sum_c ** cds
    sum_total = sum_c.sum()
    sum_w = np.reciprocal(sum_w)
    sum_c = np.reciprocal(sum_c)
    
    pmi = csr_matrix(counts)
    pmi = multiply_by_rows(pmi, sum_w)
    pmi = multiply_by_columns(pmi, sum_c)
    pmi = pmi * sum_total
    return pmi


def multiply_by_rows(matrix, row_coefs):
    normalizer = dok_matrix((len(row_coefs), len(row_coefs)))
    normalizer.setdiag(row_coefs)
    return normalizer.tocsr().dot(matrix)


def multiply_by_columns(matrix, col_coefs):
    normalizer = dok_matrix((len(col_coefs), len(col_coefs)))
    normalizer.setdiag(col_coefs)
    return matrix.dot(normalizer.tocsr())


if __name__ == '__main__':
    main()
