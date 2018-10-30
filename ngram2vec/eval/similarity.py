# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import argparse
import codecs
import numpy as np
from scipy.stats.stats import spearmanr
import sys
from utils.misc import is_word


def similarity(matrix, w2i, w1, w2, sparse=False):
    if w1 not in w2i or w2 not in w2i:
        return None
    if sparse:
        v1 = matrix[w2i[w1],:].toarray()[0]
        v2 = matrix[w2i[w2],:].toarray()[0]
    else:
        v1 = matrix[w2i[w1],:]
        v2 = matrix[w2i[w2],:]
    sim = v1.dot(v2)
    return sim


def prepare_similarities(matrix, ana_vocab, vocab, sparse=False):
    ana_matrix = matrix[[vocab["w2i"][w] if w in vocab["w2i"] else 0 for w in ana_vocab["i2w"]]]
    if sparse:
        sim_matrix = ana_matrix.dot(matrix.T).toarray()
    else:
        sim_matrix = ana_matrix.dot(matrix.T)
        sim_matrix = (sim_matrix+1)/2
    return sim_matrix
