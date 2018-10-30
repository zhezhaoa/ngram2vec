# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import argparse
import codecs
import numpy as np
from scipy.stats.stats import spearmanr
import sys
from utils.misc import is_word


def retain_words(matrix, i2w, w2i):
    i2w_word = []
    retain_index = []
    for i, w in enumerate(i2w):
        if is_word(w):
            i2w_word.append(w)
            retain_index.append(i)
    matrix = matrix[retain_index]
    w2i_word = dict([(w, i) for i, w in enumerate(i2w_word)])
    return matrix, i2w_word, w2i_word


def align_matrix(input_matrix, output_matrix, input_vocab, output_vocab):
    output_matrix_align = np.zeros(input_matrix.shape)
    for i, w in enumerate(input_vocab["i2w"]):
        if w not in output_vocab["w2i"]:
            continue
        output_matrix_align[i] = output_matrix[output_vocab["w2i"][w]]
    return output_matrix_align
