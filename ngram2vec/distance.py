# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
import argparse
import codecs
import numpy as np
from scipy.stats.stats import spearmanr
import sys
from six.moves import input
from utils.misc import normalize
from utils.matrix import load_dense, load_sparse
from eval.testset import load_analogy, get_ana_vocab
from eval.similarity import prepare_similarities
from eval.recast import retain_words, align_matrix


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--vector_file", type=str, required=True,
                        help="Path to the vector file.")
    parser.add_argument('--sparse', action='store_true',
                        help="Load sparse representation.")
    parser.add_argument('--normalize', action='store_true',
                        help="If set, vector is normalized.")
    parser.add_argument("--top_num", type=int, default=10,
                        help="The number of neighbours returned.")

    args = parser.parse_args()
    
    if args.sparse:
        matrix, vocab, _ = load_sparse(args.vector_file)
    else:
        matrix, vocab, _ = load_dense(args.vector_file)
    
    if args.normalize:
        matrix = normalize(matrix, args.sparse)
    top_num = args.top_num

    while(True):
        target = input("Enter a word (EXIT to break): ")
        if target == "EXIT":
            break
        if target not in vocab["i2w"]:
            print("Out of vocabulary")
            continue
        target_vocab = {}
        target_vocab["i2w"], target_vocab["w2i"] = [target],  {target: 0}
        sim_matrix = prepare_similarities(matrix, target_vocab, vocab, args.sparse)
        neighbours = []
        for i, w in enumerate(vocab["i2w"]):
            sim = sim_matrix[0, i]
            if target == w:
                continue
            if len(neighbours) == 0:
                neighbours.append((w, sim))
                continue
            if sim <= neighbours[-1][1] and len(neighbours) >= top_num:
                continue
            for j in range(len(neighbours)):
                if sim > neighbours[j][1]:
                    neighbours.insert(j, (w, sim))
                    break
            if len(neighbours) > top_num:
                neighbours.pop(-1)

        print("{0: <20} {1: <20}".format("word", "similarity"))
        for w, sim in neighbours:
            print("{0: <20} {1: <20}".format(w, sim))


if __name__ == '__main__':
    main()
