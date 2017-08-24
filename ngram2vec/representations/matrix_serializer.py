import numpy as np
from scipy.sparse import csr_matrix


def save_matrix(f, m):
    np.savez_compressed(f, data=m.data, indices=m.indices, indptr=m.indptr, shape=m.shape)


def load_matrix(f):
    if not f.endswith('.npz'):
        f += '.npz'
    loader = np.load(f)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def save_vocabulary(path, vocab):
    with open(path, 'w') as f:
        for w in vocab:
            print >>f, w


def load_vocabulary(path):
    with open(path) as f:
        vocab = [line.strip().split()[0] for line in f if len(line) > 0]
    return dict([(a, i) for i, a in enumerate(vocab)]), vocab


def save_count_vocabulary(path, vocab):
    with open(path, 'w') as f:
        if type(vocab) is dict:
            for w, c in vocab.iteritems():
                f.write(w+ " "+ str(c) + "\n")
        else:
            for w, c in vocab:
                f.write(w+ " "+ str(c) + "\n")   


def load_count_vocabulary(path, thr = 1):
    with open(path) as f:
        vocab = dict([(line.strip().split()[0], int(line.strip().split()[1])) for line in f if len(line) > 0 and int(line.strip().split()[1]) >= thr])
    return vocab
