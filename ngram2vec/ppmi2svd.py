# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, division
import argparse
import codecs
from sparsesvd import sparsesvd
import numpy as np
from utils.matrix import load_sparse, save_dense
from utils.vocabulary import load_vocabulary
from utils.misc import normalize


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ppmi_file", type=str, required=True,
                        help="Path to the counts (matrix) file.")
    parser.add_argument("--svd_file", type=str, required=True,
                        help="Path to the SVD file.")
    parser.add_argument("--input_vocab_file", type=str, required=True,
                        help="Path to the input vocabulary file.")
    parser.add_argument("--output_vocab_file", type=str, required=True,
                        help="Path to the output vocabulary file.")

    parser.add_argument("--size", type=int, default=100,
                        help="Vector size.")
    parser.add_argument("--normalize", action="store_true",
                        help="If set, we factorize normalized PPMI matrix")

    args = parser.parse_args()

    print("Ppmi2svd")
    input_vocab, _ = load_vocabulary(args.input_vocab_file)
    output_vocab, _ = load_vocabulary(args.output_vocab_file)
    ppmi, _, _ = load_sparse(args.ppmi_file)
    if args.normalize:
        ppmi = normalize(ppmi, sparse=True)
    ut, s, vt = sparsesvd(ppmi.tocsc(), args.size)    

    np.save(args.svd_file + ".ut.npy", ut)
    np.save(args.svd_file + ".s.npy", s)
    np.save(args.svd_file + ".vt.npy", vt)

    save_dense(args.svd_file + ".input", ut.T, input_vocab)
    save_dense(args.svd_file + ".output", vt.T, output_vocab)
    print("Ppmi2svd finished")


if __name__ == '__main__':
    main()
