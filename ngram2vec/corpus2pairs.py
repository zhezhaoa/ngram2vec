# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, division
import argparse
import multiprocessing
import codecs
import random
from math import sqrt
import six
import sys
from utils.vocabulary import load_count_vocabulary
import line2pairs
from utils.misc import is_word


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--corpus_file", type=str, required=True,
                        help="Path to the corpus file.")
    parser.add_argument("--pairs_file", type=str, required=True,
                        help="Path to the pairs file.")
    parser.add_argument("--vocab_file", type=str, required=True,
                        help="Path to the vocab file.")

    parser.add_argument("--cooccur", type=str, default="word_word",
                        choices=["word_word", "ngram_ngram"],
                        help="""Type of co-occurrence to use.
                        More types will be added. Options are
                        [word_word|ngram_ngram].""")
    parser.add_argument("--win", type=int, default=2,
                        help="Local window size.")
    parser.add_argument("--sub", type=float, default=1e-5,
                        help="Subsampling for filtering high-frequency features.")
    parser.add_argument("--processes_num", type=int, default=4,
                        help="Number of processes.")
    parser.add_argument('--dynamic_win', action='store_true',
                        help="If set, local window size is sampled from [1, win].")
    parser.add_argument('--dirty', action='store_true',
                        help="If set, removed features will be excluded before setting local window.")
    parser.add_argument('--seed', type=int, default=7,
                        help="Seed.")

    parser.add_argument("--input_order", type=int, default=1,
                        help="Order of input word-level ngram if --cooccur is set to ngram_ngram.")
    parser.add_argument("--output_order", type=int, default=2,
                        help="Order of output word-level ngram if --cooccur is set to ngram_ngram.")
    parser.add_argument('--overlap', action='store_true',
                        help="If set, overlap of ngram is allowed.")

    args = parser.parse_args()

    print("Corpus2pairs")
    processes_list = []
    # Start multiple processes.
    for i in range(0, args.processes_num):
        p = multiprocessing.Process(target=corpus2pairs_process, args=(args, i))
        p.start()
        processes_list.append(p)
    for p in processes_list:
        p.join()
    print ()
    print ("Corpus2pairs finished")


def corpus2pairs_process(args, pid):
    pairs_file = args.pairs_file + "_" + str(pid)
    pairs = codecs.open(pairs_file, "w", "utf-8")
    processes_num = args.processes_num
    sub = args.sub

    vocab = load_count_vocabulary(args.vocab_file)
    tokens_num = 0
    # Subsampling strategy.
    for w, c in six.iteritems(vocab): 
        if is_word(w):
            tokens_num += c
    sub *= tokens_num
    if sub != 0:
        subsampler = dict([(w, 1 - sqrt(sub / c)) for w, c in six.iteritems(vocab) if c > sub])
    else:
        subsampler = None
    if pid == 0:
        print("Vocabulary size: {}".format(len(vocab)))
    random.seed(args.seed)
    with codecs.open(args.corpus_file, "r", "utf-8") as f:
        lines_num = 0
        for line in f:
            lines_num += 1
            if lines_num % 1000 == 0 and pid == 0:
                print("\r{}K lines processed.".format(int(lines_num/1000)), end="")
            if lines_num % processes_num != pid:
                continue
            pairs_list = getattr(line2pairs, args.cooccur)(line, vocab, subsampler, random, args)
            for input, output in pairs_list:
                pairs.write("{} {}\n".format(input, output)) 

    pairs.close()


if __name__ == '__main__':
    main()
