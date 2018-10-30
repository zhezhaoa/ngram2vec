# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import codecs


def load_analogy(test_file):
    testset = []
    with codecs.open(test_file, "r", "utf-8") as f:
        for line in f:
            analogy = line.strip().lower().split()
            testset.append(analogy)
    return testset


def load_similarity(test_file):
    testset = []
    with codecs.open(test_file, "r", "utf-8") as f:
        for line in f:
            w1, w2, sim = line.strip().lower().split()
            testset.append(((w1, w2), float(sim)))
    return testset


def get_ana_vocab(testset):
    vocab = set()
    for analogy in testset:
        vocab.update(analogy)
    i2w = list(vocab)
    return i2w, dict([(w, i) for i, w in enumerate(i2w)])
