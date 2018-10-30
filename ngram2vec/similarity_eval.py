# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
import argparse
import codecs
import numpy as np
from scipy.stats.stats import spearmanr
import sys
from utils.misc import normalize
from utils.matrix import load_dense, load_sparse
from eval.testset import load_similarity
from eval.similarity import similarity
from eval.recast import align_matrix


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_vector_file", type=str, required=True,
                        help="Path to the input vector file.")
    parser.add_argument("--output_vector_file", type=str,
                        help="Path to the output vector file.")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the similarity task.")
    parser.add_argument('--sparse', action='store_true',
                        help="Load sparse representation.")
    parser.add_argument('--normalize', action='store_true',
                        help="If set, vector is normalized.")
    parser.add_argument("--ensemble", type=str, default="input",
                        choices=["input", "output", "add", "concat"],
                        help="""Strategies for using input/output vectors.
                        One can use only input, only output, the addition of input and output,
                        or their concatenation. Options are
                        [input|output|add|concat].""")

    args = parser.parse_args()
    
    testset = load_similarity(args.test_file)
    if args.sparse:
        matrix, vocab, _ = load_sparse(args.input_vector_file)
    else:
        matrix, vocab, _ = load_dense(args.input_vector_file)

    if not args.sparse:
        if args.ensemble == "add":
            output_matrix, output_vocab, _ = load_dense(args.output_vector_file)
            output_matrix = align_matrix(matrix, output_matrix, vocab, output_vocab)
            matrix = matrix + output_matrix
        elif args.ensemble == "concat":
            output_matrix, output_vocab, _ = load_dense(args.output_vector_file)
            output_matrix = align_matrix(matrix, output_matrix, vocab, output_vocab)
            matrix = np.concatenate([matrix, output_matrix], axis=1)
        elif args.ensemble == "output":
            matrix, vocab, _ = load_dense(args.output_vector_file)
        else: # args.ensemble == "input":
            pass

    if args.normalize:
        matrix = normalize(matrix, args.sparse)

    results = []
    for (w1, w2), sim_expected in testset:
        sim_actual = similarity(matrix, vocab["w2i"], w1, w2, args.sparse)
        if sim_actual is not None:
            results.append((sim_actual, sim_expected))
    actual, expected = zip(*results)
    print("seen/total: {}/{}".format(len(results), len(testset)))
    print("{}: {:.3f}".format(args.test_file, spearmanr(actual, expected)[0]))


if __name__ == '__main__':
    main()
