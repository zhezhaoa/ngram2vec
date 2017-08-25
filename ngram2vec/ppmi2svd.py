from sparsesvd import sparsesvd
from docopt import docopt
import numpy as np
from representations.explicit import PositiveExplicit


def main():
    args = docopt("""
    Usage:
        ppmi2svd.py [options] <ppmi> <output>
    
    Options:
        --dim NUM    Dimensionality of eigenvectors [default: 300]
        --neg NUM    Number of negative samples; subtracts its log from PMI [default: 1]
    """)
    
    ppmi_path = args['<ppmi>']
    output_path = args['<output>']
    dim = int(args['--dim'])
    neg = int(args['--neg'])
    
    explicit = PositiveExplicit(ppmi_path, normalize=False, neg=neg)

    ut, s, vt = sparsesvd(explicit.m.tocsc(), dim)

    np.save(output_path + '.ut.npy', ut)
    np.save(output_path + '.s.npy', s)
    np.save(output_path + '.vt.npy', vt)


if __name__ == '__main__':
    main()
