# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
import argparse
import codecs
from utils.vocabulary import save_count_vocabulary
import six


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pairs_file", type=str, required=True,
                        help="Path to the pairs file.")
    parser.add_argument("--input_vocab_file", type=str, required=True,
                        help="Path to the input vocabulary file.")
    parser.add_argument("--output_vocab_file", type=str, required=True,
                        help="Path to the output vocabulary file.")
    
    args = parser.parse_args()
    
    print("Pairs2vocab")
    input_vocab = {}
    output_vocab = {}
    pairs_num = 0
    with codecs.open(args.pairs_file, "r", "utf-8") as f:
        for line in f:
            pairs_num += 1
            if pairs_num % 1000**2 == 0:
                print("\r{}M pairs processed.".format(int(pairs_num/1000**2)), end="")
            pair = line.strip().split()
            if pair[0] not in input_vocab:
                input_vocab[pair[0]] = 1
            else:
                input_vocab[pair[0]] += 1      
            if pair[1] not in output_vocab:
                output_vocab[pair[1]] = 1
            else:
                output_vocab[pair[1]] += 1

    input_vocab = sorted(six.iteritems(input_vocab), key=lambda item: item[1], reverse=True)
    output_vocab = sorted(six.iteritems(output_vocab), key=lambda item: item[1], reverse=True)
    save_count_vocabulary(args.input_vocab_file, input_vocab)
    save_count_vocabulary(args.output_vocab_file, output_vocab)   
    print ("Input vocab size: {}".format(len(input_vocab)))
    print ("Output vocab size: {}".format(len(output_vocab)))
    print ("Number of pairs: {}".format(pairs_num))
    print ("Pairs2vocab finished")


if __name__ == '__main__':
    main()
