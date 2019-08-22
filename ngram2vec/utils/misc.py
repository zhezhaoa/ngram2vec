# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import numpy as np
import codecs
import six
from scipy.sparse import csr_matrix
from utils.constants import JOIN_OF_NGRAM


def get_ngram(line, pos, order):
    if pos < 0:
        return None
    if pos + order > len(line):
        return None
    ngram = line[pos]
    for i in range(1, order):
        ngram = ngram + JOIN_OF_NGRAM + line[pos + i]
    return ngram


def is_word(feature):
    return feature.isalpha()


def check_feature(feat, vocab, subsampler, random):
    if feat is None:
        return None
    if subsampler != None:
        feat = feat if feat not in subsampler or random.random() > subsampler[feat] else None
        if feat is None:
            return None
    feat = feat if feat in vocab else None
    return feat


def normalize(matrix, sparse = False):
    if sparse:
        norm = matrix.copy()
        norm.data **= 2
        norm = np.reciprocal(np.sqrt(np.array(norm.sum(axis=1))[:,0]))
        normalizer = csr_matrix((norm, (range(len(norm)), range(len(norm)))), dtype=np.float32)
        matrix = normalizer.dot(matrix)
    else:
        norm = np.sqrt(np.sum(matrix * matrix, axis=1))
        matrix = matrix / norm[:, np.newaxis]
    return matrix


def merge_vocabularies(vocab_list):
    vocab = {}
    for vocab_p in vocab_list:
        for w in vocab_p:
            if w not in vocab:
                vocab[w] = vocab_p[w]
            else:
                vocab[w] += vocab_p[w]
    return vocab

