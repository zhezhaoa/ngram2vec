# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
import argparse
import subprocess
import os


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--counts_file", type=str, required=True,
                        help="Path to the counts file.")
    parser.add_argument("--input_vocab_file", type=str, required=True,
                        help="Path to the input vocabulary file.")
    parser.add_argument("--output_vocab_file", type=str, required=True,
                        help="Path to the output vocabulary file.")
    parser.add_argument("--input_vector_file", type=str, required=True,
                        help="Path to the input vector file.")
    parser.add_argument("--output_vector_file", type=str, required=True,
                        help="Path to the output vector file.")

    parser.add_argument("--size", type=int, default=100,
                        help="")
    parser.add_argument("--threads_num", type=int, default=4,
                        help="")
    parser.add_argument("--iter", type=int, default=10,
                        help="")

    args = parser.parse_args()

    print("Counts2glove")
    command = ["./glove/glove"]
    command.extend(["--counts_file", args.counts_file])
    command.extend(["--input_vocab_file", args.input_vocab_file])
    command.extend(["--output_vocab_file", args.output_vocab_file])
    command.extend(["--input_vector_file", args.input_vector_file])
    command.extend(["--output_vector_file", args.output_vector_file])

    command.extend(["--size", str(args.size)])
    command.extend(["--threads_num", str(args.threads_num)])
    command.extend(["--iter", str(args.iter)])
    
    return_code = subprocess.call(command)
    print()
    print("Counts2glove finished")


if __name__ == '__main__':
    main()
