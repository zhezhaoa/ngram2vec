import heapq

from scipy.sparse import dok_matrix, csr_matrix
import numpy as np

from representations.matrix_serializer import load_vocabulary, load_matrix


class Explicit:
    """
    Base class for explicit representations. Assumes that the serialized input is e^PMI.
    """
    
    def __init__(self, path, normalize=True):
        self.wi, self.iw = load_vocabulary(path + '../words.vocab')
        self.ci, self.ic = load_vocabulary(path + '../contexts.vocab')
        self.m = load_matrix(path + 'ppmi')
        self.m.data = np.log(self.m.data)
        self.normal = normalize
        if normalize:
            self.normalize()
    
    def normalize(self):
        m2 = self.m.copy()
        m2.data **= 2
        norm = np.reciprocal(np.sqrt(np.array(m2.sum(axis=1))[:, 0]))
        normalizer = dok_matrix((len(norm), len(norm)))
        normalizer.setdiag(norm)
        self.m = normalizer.tocsr().dot(self.m)
    
    def represent(self, w):
        if w in self.wi:
            return self.m[self.wi[w], :]
        else:
            return csr_matrix((1, len(self.ic)))
    
    def similarity_first_order(self, w, c):
        return self.m[self.wi[w], self.ci[c]]
    
    def similarity(self, w1, w2):
        """
        Assumes the vectors have been normalized.
        """
        if w1 not in self.wi or w2 not in self.wi :
            return None
        return self.represent(w1).dot(self.represent(w2).T)[0, 0]
    
    def closest_contexts(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.represent(w)
        return heapq.nlargest(n, zip(scores.data, [self.ic[i] for i in scores.indices]))
    
    def closest(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.m.dot(self.represent(w).T).T.tocsr()
        return heapq.nlargest(n, zip(scores.data, [self.iw[i] for i in scores.indices]))


class PositiveExplicit(Explicit):
    """
    Positive PMI (PPMI) with negative sampling (neg).
    Negative samples shift the PMI matrix before truncation.
    """
    
    def __init__(self, path, normalize=True, neg=1):
        Explicit.__init__(self, path, False)
        self.m.data -= np.log(neg)
        self.m.data[self.m.data < 0] = 0
        self.m.eliminate_zeros()
        if normalize:
            self.normalize()
