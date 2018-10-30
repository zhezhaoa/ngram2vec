# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import numpy as np
from scipy.sparse import csr_matrix
import codecs


def save_dense(path, matrix, vocab):
    with codecs.open(path, "w", "utf-8") as f:
        line = str(matrix.shape[0]) + " " + str(matrix.shape[1]) + "\n"
        f.write(line)
        for i in range(len(vocab)):
            line = " ".join([str(v) for v in matrix[i, :]])
            line = vocab[i] + " " + line + "\n"
            f.write(line)


def load_dense(path):
    vocab_size, size = 0, 0
    vocab = {}
    vocab["i2w"], vocab["w2i"] = [], {}
    with codecs.open(path, "r", "utf-8") as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                vocab_size = int(line.strip().split()[0])
                size = int(line.strip().split()[1])
                matrix = np.zeros(shape=(vocab_size, size), dtype=np.float32)
                continue
            vec = line.strip().split()
            vocab["i2w"].append(vec[0])
            matrix[len(vocab["i2w"])-1, :] = np.array([float(x) for x in vec[1:]])
    for i, w in enumerate(vocab["i2w"]):
        vocab["w2i"][w] = i
    return matrix, vocab, size


def save_sparse(path, matrix, vocab):
     with codecs.open(path, "w", "utf-8") as f:
        line = str(matrix.get_shape()[0]) + " " + str(matrix.get_shape()[1]) + "\n"
        f.write(line)
        for i in range(len(vocab)):
            ind = matrix.indices[matrix.indptr[i]: matrix.indptr[i+1]]
            dat = matrix.data[matrix.indptr[i]: matrix.indptr[i+1]]
            line = [str(a1)+":"+str(a2) for a1, a2 in zip(ind, dat)]
            line = " ".join(line)
            line = vocab[i] + " " + line + "\n"
            f.write(line)


def load_sparse(path):
    vocab_size, size = 0, 0
    vocab = {}
    vocab["i2w"], vocab["w2i"] = [], {}
    row, col, data = [], [], []
    with codecs.open(path, "r", "utf-8") as f:
        first_line = True
        lines_num = 0
        for line in f:
            if first_line:
                first_line = False
                vocab_size = int(line.strip().split()[0])
                size = int(line.strip().split()[1])
                continue
            line = line.strip().split()
            vocab["i2w"].append(line[0])
            vector = line[1:]
            for v in vector:
                row.append(lines_num)
                col.append(int(v.split(":")[0]))
                data.append(float(v.split(":")[1]))
            lines_num += 1
        for i, w in enumerate(vocab["i2w"]):
            vocab["w2i"][w] = i
        row, col, data = np.array(row), np.array(col), np.array(data)
        matrix = csr_matrix((data, (row, col)), shape=(vocab_size, size))
    return matrix, vocab, size
