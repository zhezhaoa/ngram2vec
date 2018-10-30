# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, division
import codecs
import argparse
from collections import Counter
from math import sqrt
from sys import getsizeof
import numpy as np
import sys
import cPickle as pickle
import os
from utils.vocabulary import load_vocabulary


def aggregate(tmpcounts, input, output, memory_size_used, args):
    if args.aggregate == "stripes":
        if input in tmpcounts:
            tmp_size = getsizeof(tmpcounts[input])
            tmpcounts[input].update({output: 1})
            memory_size_used += getsizeof(tmpcounts[input]) - tmp_size
        else:
            tmpcounts[input] = Counter({output: 1})
            memory_size_used += getsizeof(tmpcounts[input])
    else: # args.aggregate == "pairs"
        if (input, output) in tmpcounts:
            tmpcounts[(input, output)] += 1
        else:
            tmpcounts[(input, output)] = 1
            memory_size_used += getsizeof(int(0))
    return memory_size_used


def write_tmpfiles(tmpcounts, tmpfile_id, args):
    with codecs.open("{}_{}".format(args.counts_file, tmpfile_id), 'wb') as f:
        if args.aggregate == "stripes":
            sorted_input = sorted(tmpcounts.keys())
            for i in sorted_input:
                pickle.dump((i, tmpcounts[i]), f)
        else: # args.aggregate == "pairs"
            sorted_pairs = sorted(tmpcounts.keys())
            for input, output in sorted_pairs:
                pickle.dump(((input, output), tmpcounts[(input, output)]), f)


def merge(out, buffer_min, args):
    if args.aggregate == "stripes":
        out[1].update(buffer_min[1])
    else: # args.aggregate == "pairs"
        out[1] += buffer_min[1]
    return out


def write_buffer(counts, out, counts_num, input_i2w, output_i2w, args):
    if args.aggregate == "stripes":
        sorted_output = sorted(out[1].keys())
        for w in sorted_output:
            counts_num += 1
            if counts_num % 1000**2 == 0:
                print("\r{}M counts processed.".format(int(counts_num/1000**2)), end="")
            if args.output_id:
                counts.write("{} {} {}\n".format(out[0], w, out[1][w]))
            else:
                counts.write("{} {} {}\n".format(input_i2w[out[0]], output_i2w[w], out[1][w]))
    else: # args.aggregate == "pairs"
        counts_num += 1
        if counts_num % 1000**2 == 0:
            print("\r{}M counts processed.".format(int(counts_num/1000**2)), end="")
        if args.output_id:
            counts.write("{} {} {}\n".format(out[0][0], out[0][1], out[1]))
        else:
            counts.write("{} {} {}\n".format(input_i2w[out[0][0]], output_i2w[out[0][1]], out[1]))
    return counts_num


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pairs_file", type=str, required=True,
                        help="Path to the pairs file.")
    parser.add_argument("--input_vocab_file", type=str, required=True,
                        help="Path to the input vocabulary file.")
    parser.add_argument("--output_vocab_file", type=str, required=True,
                        help="Path to the output vocabulary file.")
    parser.add_argument("--counts_file", type=str, required=True,
                        help="Path to the counts (matrix) file.")
    parser.add_argument('--output_id', action='store_true',
                        help="If set, output (id id count) instead of (word word count).")
    parser.add_argument("--memory_size", type=float, default=4.0,
                        help="Memory size.")
    parser.add_argument("--aggregate", type=str, default="stripes",
                        choices=["pairs", "stripes"],
                        help="""Different strategies of building counts (matrix).
                        Options are
                        [pairs|stripes].""")
    
    args = parser.parse_args()

    print("Pairs2counts")

    input_vocab, output_vocab = {}, {}
    input_vocab["i2w"], input_vocab["w2i"] =load_vocabulary(args.input_vocab_file)
    output_vocab["i2w"], output_vocab["w2i"] = load_vocabulary(args.output_vocab_file)
    memory_size = args.memory_size * 1000**3
    tmpcounts = {} #store co-occurrence matrix in dictionary
    tmpfiles_num = 0
    memory_size_used = 0
    pairs_num = 0
    with codecs.open(args.pairs_file, "r", "utf-8") as f:
        for line in f:
            pairs_num += 1
            if pairs_num % 1000**2 == 0:
                print("\r{}M pairs processed.".format(int(pairs_num/1000**2)), end="")
            if getsizeof(tmpcounts) + memory_size_used > memory_size * 0.8:
                write_tmpfiles(tmpcounts, tmpfiles_num, args)
                tmpfiles_num += 1
                tmpcounts.clear()
                memory_size_used = 0
            pair = line.strip().split()
            input = input_vocab["w2i"][pair[0]]
            output = output_vocab["w2i"][pair[1]]
            memory_size_used = aggregate(tmpcounts, input, output, memory_size_used, args)
    write_tmpfiles(tmpcounts, tmpfiles_num, args)
    tmpcounts.clear()
    tmpfiles_num += 1

    print()
    tmpfiles = []
    top_buffer = []
    counts_num = 0
    counts = codecs.open(args.counts_file, "w", "utf-8")
    for i in range(tmpfiles_num):
        tmpfiles.append(codecs.open("{}_{}".format(args.counts_file, i), "rb"))
        top_buffer.append(pickle.load(tmpfiles[i]))
    top_buffer_keys = [c[0] for c in top_buffer]
    min_index = top_buffer_keys.index(min(top_buffer_keys))
    out = list(top_buffer[min_index])
    top_buffer[min_index] = pickle.load(tmpfiles[min_index])

    while True:
        top_buffer_keys = [c[0] for c in top_buffer]
        min_index = top_buffer_keys.index(min(top_buffer_keys))
        if top_buffer[min_index][0] == out[0]:
            out = merge(out, top_buffer[min_index], args)
        else:
            counts_num = write_buffer(counts, out, counts_num, input_vocab["i2w"], output_vocab["i2w"], args)
            out = list(top_buffer[min_index])
        try:
            top_buffer[min_index] = pickle.load(tmpfiles[min_index])
        except Exception:
            if args.aggregate == "stripes":
                top_buffer[min_index] = (sys.maxint, Counter())
            else: # args.aggregate == "pairs"
                top_buffer[min_index] = ((sys.maxint, sys.maxint), 0)
            tmpfiles_num -= 1
        if tmpfiles_num == 0:
            counts_num = write_buffer(counts, out, counts_num, input_vocab["i2w"], output_vocab["i2w"], args)
            break
    counts.close()
    print("Number of counts: {}".format(counts_num))
    for i in range(len(top_buffer)):
        os.remove("{}_{}".format(args.counts_file, i))

    print("Pairs2counts finished")


if __name__ == '__main__':
    main()
