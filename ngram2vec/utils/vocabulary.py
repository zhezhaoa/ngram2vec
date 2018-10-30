# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
import codecs
import six


def save_vocabulary(path, vocab):
    with codecs.open(path, "w", "utf-8") as f:
        for w in vocab:
            f.write("{}\n".format(w))


def load_vocabulary(path):
    with codecs.open(path, "r", "utf-8") as f:
        i2w = [line.strip().split()[0] for line in f if len(line) > 0]
    return i2w, dict([(w, i) for i, w in enumerate(i2w)])


def save_count_vocabulary(path, vocab):
    with codecs.open(path, "w", "utf-8") as f:
        if isinstance(vocab, dict):
            for w, c in six.iteritems(vocab):
                f.write("{} {}\n".format(w, c))
        else: # isinstance(vocab, list) == True
            for w, c in vocab:
                f.write("{} {}\n".format(w, c))


def load_count_vocabulary(path, thr=1):
    with codecs.open(path, "r", "utf-8") as f:
        vocab = [line.strip().split() for line in f if len(line) > 0]
        vocab = dict([(w, int(c)) for w, c in vocab if int(c) >= thr])
    return vocab
