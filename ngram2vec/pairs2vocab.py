# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
import argparse
import codecs
from multiprocessing import Pool
from utils.vocabulary import save_count_vocabulary
from utils.misc import merge_vocabularies
import six


def pairs2vocab_process(pairs_file, proc_id, proc_num, args):
    """ 
    .
    """
    input_vocab = {}
    output_vocab = {}
    pairs_num = 0
    with open(pairs_file, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if (line_id % proc_num) == proc_id:
                if proc_id == 0:
                    if pairs_num % 1000 == 0:
                        print("\r{}M pairs processed.".format(int(pairs_num*proc_num/1000**2)), end="")
                pair = line.strip().split()
                if pair[0] not in input_vocab:
                    input_vocab[pair[0]] = 1
                else:
                    input_vocab[pair[0]] += 1      
                if pair[1] not in output_vocab:
                    output_vocab[pair[1]] = 1
                else:
                    output_vocab[pair[1]] += 1
                pairs_num += 1
    return (input_vocab, output_vocab)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pairs_file", type=str, required=True,
                        help="Path to the pairs file.")
    parser.add_argument("--input_vocab_file", type=str, required=True,
                        help="Path to the input vocabulary file.")
    parser.add_argument("--output_vocab_file", type=str, required=True,
                        help="Path to the output vocabulary file.")

    parser.add_argument("--processes_num", type=int, default=4,
                        help="Number of processes.")
    
    args = parser.parse_args()
    
    print("Pairs2vocab")

    pool = Pool(args.processes_num)
    vocab_list = []
    for i in range(args.processes_num):
        vocab_list.append((pool.apply_async(func=pairs2vocab_process, args=[args.pairs_file, i, args.processes_num, args])))
    
    pool.close()
    pool.join()

    input_vocab_list = [v.get()[0] for v in vocab_list]
    input_vocab = merge_vocabularies(input_vocab_list)

    output_vocab_list = [v.get()[1] for v in vocab_list]
    output_vocab = merge_vocabularies(output_vocab_list)

    input_vocab = sorted(six.iteritems(input_vocab), key=lambda item: item[1], reverse=True)
    output_vocab = sorted(six.iteritems(output_vocab), key=lambda item: item[1], reverse=True)
    save_count_vocabulary(args.input_vocab_file, input_vocab)
    save_count_vocabulary(args.output_vocab_file, output_vocab)   
    print ("Input vocab size: {}".format(len(input_vocab)))
    print ("Output vocab size: {}".format(len(output_vocab)))
    print ("Pairs2vocab finished")


if __name__ == '__main__':
    main()

