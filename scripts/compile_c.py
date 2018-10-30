# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
import subprocess
import os


def compile(source, target):
    CC = "gcc"
    CFLAGS = "-lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result"
    command = [CC, source, "-o", target]
    command.extend(CFLAGS.split())
    print("Compilation command: " + " ".join(command))
    return_code = subprocess.call(command)
    if return_code > 0:
        exit(return_code)

def main():
    # Word2vec toolkit.
    source = os.path.join("word2vec", "word2vec.c")
    target = os.path.join("word2vec", "word2vec")
    compile(source, target)
    # GloVe toolkit.
    source = os.path.join("glove", "glove.c")
    target = os.path.join("glove", "glove")
    compile(source, target)


if __name__ == '__main__':
    main()
