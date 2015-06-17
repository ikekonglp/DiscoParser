import argparse
import codecs
import os
import re
import sys
import h5py
import numpy as np


if __name__ == '__main__':
    f = h5py.File("../torch/result.hd5", "r")
    myarr = f['dev_result'][()] # [()] means to read the entire array in
    f.close()

    results = [int(x) for x in myarr]
    results.append(1)

    f_dev = open("dev_imp", "r")
    dev = f_dev.readlines()
    f_dev.close()

    f = h5py.File("mr.hdf5", "r")
    myarr = f['dev_label'][()]
    f.close()
    
    gold = [x for x in myarr]
    
    correct = 0.0
    total = 0.0
    assert(len(gold) == len(results) and len(gold) == len(dev))

    relation_dict = {}
    max_index = -1
    f_label_table = open("../preprocess_script/relation_table")
    for line in f_label_table:
        args = line.split("\t")
        max_index =  int(args[1].strip()) if int(args[1].strip()) > max_index else max_index
        relation_dict[int(args[1].strip())] = args[0].strip()
    f_label_table.close()
    size = max_index + 1

    cm = np.zeros((size,size))

    i = 0
    while i < len(gold):
        correct_label_set = [gold[i]]
        content = dev[i].split("||||")[:2]

        # proceed to all the possible labels
        while (i+1) < len(gold):
            if dev[i+1].split("||||")[:2] == content:
                # print content
                # print dev[i+1].split("||||")[:2]
                correct_label_set.append(gold[i+1])
                i += 1
            else:
                break

        if results[i] in correct_label_set:
            correct += 1
            cm[int(results[i])][int(results[i])] += 1
        else:
            cm[int(correct_label_set[0])][int(results[i])] += 1
        total += 1
        i += 1
    f_out = open("ana_table", "w")
    # print first line
    f_out.write(" \t")
    f_out.write("\t".join([relation_dict[y] if y in relation_dict else str(y) for y in xrange(0, size)]))
    f_out.write("\n")

    for ix in xrange(0, size):
        f_out.write(relation_dict[ix] if ix in relation_dict else str(ix))
        for iy in xrange(0, size):
            f_out.write("\t" + (str(int(cm[ix][iy])) if ix != iy else ("*" + str(int(cm[ix][iy])))))
        f_out.write("\n")

    print correct/total
    f_out.close()