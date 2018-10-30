# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, division
import argparse
import codecs
import os
import random
from sys import getsizeof


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the original file.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the shuffled file.")
    parser.add_argument("--memory_size", type=float, default=4.0,
                        help="Memory size. Shuffle and write out data when memory is used up.")
    
    args = parser.parse_args()

    memory_size = args.memory_size * 1000**3
    memory_size_used = 0
    lines = []
    lines_num_per_file = []
    tmpfiles_num = 0
    lines_num = 0
    # Shuffle step 1.
    with codecs.open(args.input_file, "r", "utf-8") as f:
        for line in f:
            lines_num += 1
            if lines_num % 1000 == 0:
                print("\r{}M lines processed.".format(int(lines_num/1000**2)), end="")
            lines.append(line)
            memory_size_used += getsizeof(line)
            if getsizeof(lines) + memory_size_used > memory_size * 0.8:
                random.shuffle(lines)
                with codecs.open(args.output_file + str(tmpfiles_num), "w", "utf-8") as f:
                    for l in lines:
                        f.write(l)
                if len(lines_num_per_file) == 0:
                    lines_num_per_file.append(lines_num)
                else:
                    lines_num_per_file.append(lines_num-lines_num_per_file[-1])
                lines = []
                tmpfiles_num += 1
                memory_size_used = 0

    random.shuffle(lines)
    with codecs.open(args.output_file + str(tmpfiles_num), "w", "utf-8") as f:
        for l in lines:
            f.write(l)
        if len(lines_num_per_file) == 0:
            lines_num_per_file.append(lines_num)
        else:
            lines_num_per_file.append(lines_num-lines_num_per_file[-1])
        lines = []
        tmpfiles_num += 1
    print()
    print("Number of tmpfiles: {}".format(tmpfiles_num))


    # Shuffle step 2.
    lines_num = 0
    output = codecs.open(args.output_file, "w", "utf-8")
    tmpfiles = []
    for i in range(tmpfiles_num):
        tmpfiles.append(codecs.open(args.output_file + str(i), "r", "utf-8"))
    
    limit = int(lines_num_per_file[0] / tmpfiles_num)
    for i in range(tmpfiles_num-1):
        lines = []
        for f in tmpfiles:
            for j in range(limit):
                line = f.readline()
                if len(line) > 0:
                    lines_num += 1
                    lines.append(line)
                    if lines_num % 1000 == 0:
                        print("\r{}M lines processed.".format(int(lines_num/1000**2)), end="")
        random.shuffle(lines)
        for line in lines:
            output.write(line)
    lines = []
    for f in tmpfiles:
        for line in f:
            lines_num += 1
            if lines_num % 1000 == 0:
                print("\r{}M lines processed.".format(int(lines_num/1000**2)), end="")        
            lines.append(line)
    random.shuffle(lines)
    for line in lines:
        output.write(line)

    for i in range(tmpfiles_num):
        tmpfiles[i].close()
    for i in range(tmpfiles_num):
        os.remove(args.output_file + str(i))
    output.close()
    print()
    print("Number of lines: {}".format(lines_num))


if __name__ == '__main__':
    main()
