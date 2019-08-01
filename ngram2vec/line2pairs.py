# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, division
from utils.misc import get_ngram, check_feature


def word_word(line, vocab, subsampler, random, args):
    pairs_list = []
    if args.dynamic_win:
        win = random.randint(1, args.win)
    else:
        win = args.win
    line = [t if t in vocab else None for t in line.strip().split()]
    if subsampler:
        line = [t if t not in subsampler or random.random() > subsampler[t] else None for t in line]
    if args.dirty:
        line = [t for t in line if t is not None]
    for i in range(len(line)):
        input = get_ngram(line, i, 1)
        if input is None:
            continue
        start, end = i - win, i + win
        for j in range(start, end + 1):
            if i == j:
                continue
            output = get_ngram(line, j, 1)
            if output is None:
                continue
            pairs_list.append((input, output))
    return pairs_list


def ngram_ngram(line, vocab, subsampler, random, args):
    pairs_list = []
    if args.dynamic_win:
        win = random.randint(1, args.win)
    else:
        win = args.win
    input_order, output_order = args.input_order, args.output_order
    overlap = args.overlap
    tokens = line.strip().split()
    for i in range(len(tokens)):
        for j in range(1, input_order+1):
            input = get_ngram(tokens, i, j)
            input = check_feature(input, vocab, subsampler, random)
            if input is None:
                continue
            for k in range(1, output_order+1):
                start = i - win + j - 1
                end = i + win - k + 1
                for l in range(start, end + 1):
                    if overlap:
                        if i == l and j == k:
                            continue
                    else:
                        if len(set(range(i, i + j)) & set(range(l, l + k))) > 0:
                            continue
                    output = get_ngram(tokens, l, k)
                    output = check_feature(output, vocab, subsampler, random)
                    if output is None:
                        continue
                    pairs_list.append((input, output))
    return pairs_list
