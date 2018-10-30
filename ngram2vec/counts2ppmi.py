# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, division
import argparse
import codecs
from scipy.sparse import csr_matrix
import numpy as np
from utils.vocabulary import load_vocabulary
from utils.matrix import save_sparse
from utils.misc import normalize


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--counts_file", type=str, required=True,
                        help="Path to the counts (matrix) file.")
    parser.add_argument("--input_vocab_file", type=str, required=True,
                        help="Path to the input vocabulary file.")
    parser.add_argument("--output_vocab_file", type=str, required=True,
                        help="Path to the output vocabulary file.")
    parser.add_argument("--ppmi_file", type=str, required=True,
                        help="Path to the PPMI file.")

    parser.add_argument("--cds", type=float, default=1.0,
                        help="Context distribution smoothing.")
    parser.add_argument("--neg", type=float, default=1.0,
                        help="Negative sampling, shifted value on PPMI.")
    
    args = parser.parse_args()

    print("Counts2ppmi")
    input_vocab = {}
    output_vocab = {}
    input_vocab["i2w"], input_vocab["w2i"] = load_vocabulary(args.input_vocab_file)
    output_vocab["i2w"], output_vocab["w2i"] = load_vocabulary(args.output_vocab_file)
    matrix = load_sparse_from_counts(args.counts_file, input_vocab["w2i"], output_vocab["w2i"], is_id=True)
    pmi = calc_pmi(matrix, args.cds)

    # From PMI to Shifted PPMI (SPPMI).
    pmi.data = np.log(pmi.data)
    pmi.data -= np.log(args.neg)
    pmi.data[pmi.data < 0] = 0
    pmi.eliminate_zeros()
    ppmi = pmi
    # Save PPMI in txt format.
    save_sparse(args.ppmi_file, ppmi, input_vocab["i2w"])
    print()
    print("Counts2ppmi finished")


def load_sparse_from_counts(counts_file, input_vocab, output_vocab, is_id=False):
    counts_num = 0
    row, col, data = [], [], []
    with codecs.open(counts_file, "r", "utf-8") as f:
        for line in f:
            counts_num += 1
            if counts_num % 1000**2 == 0:
                print("\r{}M counts processed.".format(int(counts_num/1000**2)), end="")
            input, output, count = line.strip().split()
            if is_id:
                row.append(int(input))
                col.append(int(output))
            else:
                row.append(input_vocab[input])
                col.append(output_vocab[output])
            data.append(float(count))
    matrix = csr_matrix((data, (row, col)), shape=(len(input_vocab), len(output_vocab)), dtype=np.float32)
    return matrix


def calc_pmi(matrix, cds):
    row_sum = np.array(matrix.sum(axis=1))[:, 0]
    col_sum = np.array(matrix.sum(axis=0))[0, :]
    if cds != 1:
        col_sum = col_sum ** cds
    sum_total = col_sum.sum()

    pmi = multiply_by_rows(matrix, np.reciprocal(row_sum))
    pmi = multiply_by_columns(pmi, np.reciprocal(col_sum))
    pmi = pmi * sum_total
    return pmi


def multiply_by_rows(matrix, row_sum):
    normalizer = csr_matrix((row_sum, (range(len(row_sum)), range(len(row_sum)))), dtype=np.float32)
    return normalizer.dot(matrix)


def multiply_by_columns(matrix, col_sum):
    normalizer = csr_matrix((col_sum, (range(len(col_sum)), range(len(col_sum)))), dtype=np.float32)
    return matrix.dot(normalizer)


if __name__ == '__main__':
    main()
