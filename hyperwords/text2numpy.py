from docopt import docopt
import numpy as np

from representations.matrix_serializer import save_vocabulary


def main():
    args = docopt("""
    Usage:
        text2numpy.py <path>
    """)
    
    path = args['<path>']
    
    matrix = read_vectors(path)
    iw = sorted(matrix.keys())
    
    new_matrix = np.zeros(shape=(len(iw), len(matrix[iw[0]])), dtype=np.float32)
    for i, word in enumerate(iw):
        if word in matrix:
            new_matrix[i, :] = matrix[word]
    
    np.save(path + '.npy', new_matrix)
    save_vocabulary(path + '.vocab', iw)


def read_vectors(path):
    vectors = {}
    with open(path) as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            tokens = line.strip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
    return vectors


if __name__ == '__main__':
    main()
