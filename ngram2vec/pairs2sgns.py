# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
import argparse
import subprocess
import os


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pairs_file", type=str, required=True,
                        help="Path to the pairs file.")
    parser.add_argument("--input_vocab_file", type=str, required=True,
                        help="Path to the input vocabulary file.")
    parser.add_argument("--output_vocab_file", type=str, required=True,
                        help="Path to the output vocabulary file.")
    parser.add_argument("--input_vector_file", type=str, required=True,
                        help="Path to the input vector file.")
    parser.add_argument("--output_vector_file", type=str, required=True,
                        help="Path to the output vector file.")

    parser.add_argument("--negative", type=int, default=5,
                        help="")
    parser.add_argument("--size", type=int, default=100,
                        help="")
    parser.add_argument("--threads_num", type=int, default=4,
                        help="")
    parser.add_argument("--iter", type=int, default=3,
                        help="")

    args = parser.parse_args()

    print("Pairs2sgns")
    command = ["./word2vec/word2vec"]
    command.extend(["--pairs_file", args.pairs_file])
    command.extend(["--input_vocab_file", args.input_vocab_file])
    command.extend(["--output_vocab_file", args.output_vocab_file])
    command.extend(["--input_vector_file", args.input_vector_file])
    command.extend(["--output_vector_file", args.output_vector_file])

    command.extend(["--negative", str(args.negative)])
    command.extend(["--size", str(args.size)])
    command.extend(["--threads_num", str(args.threads_num)])
    command.extend(["--iter", str(args.iter)])

    return_code = subprocess.call(command)
    print()
    print("Pairs2sgns finished")


if __name__ == '__main__':
    main()
