import argparse
import codecs
import os
import re
import sys
from collections import defaultdict
from collections import Counter
import nltk
import h5py
import numpy as np

MAX_LENGTH = 100

class IndexDict:
    def __init__(self):
        self.dict = {}
        # save 1 for padding and 2 for the OOV
        self.ind = 3

    def get_or_add(self, word):
        if word in self.dict:
            return self.dict[word]
        else:
            self.dict[word] = self.ind
            self.ind += 1
            return self.ind-1

    def get(self, word):
        if word in self.dict:
            return self.dict[word]
        else:
            return 2

    def length(self):
        return (self.ind-1)

def gen_representation(dataset, vol, relations, throw_away=False):
    targ1 = []
    targ2 = []
    trelation = []
    for line in dataset:
        args = line.split("||||")
        tokens_arg1 = [word.strip() for word in nltk.word_tokenize(args[0])]
        tokens_arg2 = [word.strip() for word in nltk.word_tokenize(args[1])]
        relation = args[2].strip()

        num_arg1 = [vol.get(word) for word in tokens_arg1]
        num_arg2 = [vol.get(word) for word in tokens_arg2]

        if len(num_arg1)==0 or len(num_arg2)==0:
            print "let me know"
            print tokens_arg1, tokens_arg2, relation
            exit()


        arg1 = []
        for num in num_arg1:
            arg1.append(num)

        while len(arg1) < MAX_LENGTH:
            arg1.insert(0, 1)

        arg2 = []
        for num in num_arg2:
            arg2.append(num)

        while len(arg2) < MAX_LENGTH:
            arg2.insert(0, 1)

        if (throw_away) and (len(arg2) > MAX_LENGTH or len(arg1) > MAX_LENGTH):
            continue 


        targ1.append(arg1)
        targ2.append(arg2)

        trelation.append(relations.get(relation))

        assert(len(arg1) == MAX_LENGTH)
        assert(len(arg2) == MAX_LENGTH)
        assert(len(targ1) == len(trelation))
        
        #break
    # print [targ1, targ2, trelation]
    return [targ1, targ2, trelation]

def load_word2vec(vol):
    verts = [None] * vol.length()
    word2vec = {}
    f = open('../dataset/GoogleNews-vectors-negative300.readable', 'r')
    for line in f:
        args = line.split()
        if len(args) == 301:
            if args[0] in vol.dict:
                fnums = args[1:]
                v = [float(x) for x in fnums]
                print args[0]
                verts[vol.get(args[0])-1] = np.array(v, dtype=float)
    verts[0] = np.array([0] * 300, dtype=float)
    for i in xrange(0, len(verts)):
        if verts[i] is None:
            verts[i] = np.array(np.random.normal(0, 0.01, size=300), dtype=float)
    f.close()
    return verts

if __name__ == '__main__':
    f_train = open("train_imp", "r")
    f_dev = open("dev_imp", "r")
    f_test = open("test_imp", "r")

    train = f_train.readlines()
    dev = f_dev.readlines()
    test = f_test.readlines()
    
    cnt = Counter()
    relations = IndexDict()
    vol = IndexDict()

    for line in train:
        args = line.split("||||")

        tokens_arg1 = [word.strip() for word in nltk.word_tokenize(args[0])]
        tokens_arg2 = [word.strip() for word in nltk.word_tokenize(args[1])]
        relation = args[2].strip()

        for w in tokens_arg1:
            cnt[w] += 1
        for w in tokens_arg2:
            cnt[w] += 1
        relations.get_or_add(relation)

    for w in cnt:
        if cnt[w] > 1:
            vol.get_or_add(w)

    verts = load_word2vec(vol)

    # for word in vol.dict:
    #     print word, vol.get(word)

    f = h5py.File("mr.hdf5", "w")
    f['train_arg1'], f['train_arg2'], f['train_label'] = ([np.array(ele, dtype=int) for ele in gen_representation(train, vol, relations, True)])
    f['dev_arg1'], f['dev_arg2'], f['dev_label'] = ([np.array(ele, dtype=int) for ele in gen_representation(dev, vol, relations)])
    f['test_arg1'], f['test_arg2'], f['test_label'] = ([np.array(ele, dtype=int) for ele in gen_representation(test, vol, relations)])
    f['embeding'] = np.array(verts)
    print np.array(verts).shape
    f_train.close()
    f_dev.close()
    f_test.close()