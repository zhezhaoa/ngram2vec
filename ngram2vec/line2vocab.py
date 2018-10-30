# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
from collections import Counter
from utils.constants import JOIN_OF_NGRAM
from utils.misc import get_ngram


def word(line, args):
    return Counter(line.strip().split())


def ngram(line, args):
    order = args.order
    line = line.strip().split()
    vocab = Counter()
    for i in range(len(line)):
        for j in range(1, order+1):
            ngram = get_ngram(line, i, j)
            if ngram is None:
                continue
            vocab.update([ngram])
    return vocab
