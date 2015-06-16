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
from nltk.parse import stanford
from nltk import Tree
import pickle

STANFORD_DIR = "/home/lingpenk/research/parsers/stanford/stanford330/"

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

def gen_rules(tree):
    small_rule_cnt = Counter()
    for pos in tree.treepositions(order='postorder'):
        st = tree[pos]
        if isinstance(st, str) or isinstance(st, unicode):
            continue
        else:
            if st.height() == 2:
            # preterminals also don't generate rules
                continue
            rule_list = [str(n.label()) if isinstance(n, Tree) else n for n in st]
            rule_list.insert(0, str(st.label()))
            rule_str = " ".join(rule_list)
            small_rule_cnt[rule_str] += 1
    return small_rule_cnt

def gen_vector(rule_cnt, rule_index_dict):
    vec = [0] * (rule_index_dict.length())
    for r in rule_cnt:
        vec[(rule_index_dict.get(r)-1)] = rule_cnt[r]
    return vec




if __name__ == '__main__':
    # f_train = open("train_imp", "r")
    # f_dev = open("dev_imp", "r")
    # f_test = open("test_imp", "r")

    # train = f_train.readlines()
    # dev = f_dev.readlines()
    # test = f_test.readlines()
    
    # cnt = Counter()
    # relations = IndexDict()
    # vol = IndexDict()

    # os.environ['STANFORD_PARSER'] = STANFORD_DIR + "stanford-parser.jar"
    # os.environ['STANFORD_MODELS'] = STANFORD_DIR + "stanford-parser-3.3.0-models.jar"

    # parser = stanford.StanfordParser(model_path= STANFORD_DIR + "englishPCFG.ser.gz")

    # arg1s = []
    # arg2s = []
    # # ind = 0
    # for line in train:
    #     # ind += 1
    #     # print ind
    #     args = line.split("||||")
    #     arg1s.append(args[0].strip().encode('utf-8'))
    #     arg2s.append(args[1].strip().encode('utf-8'))

    # print len(arg1s), len(arg2s)
    # train_parse_arg1s = parser.raw_parse_sents(arg1s, True)
    # train_parse_arg2s = parser.raw_parse_sents(arg2s, True)

    # arg1s = []
    # arg2s = []
    # # ind = 0
    # for line in dev:
    #     # ind += 1
    #     # print ind
    #     args = line.split("||||")
    #     arg1s.append(args[0].strip().encode('utf-8'))
    #     arg2s.append(args[1].strip().encode('utf-8'))

    # print len(arg1s), len(arg2s)
    # dev_parse_arg1s = parser.raw_parse_sents(arg1s, True)
    # dev_parse_arg2s = parser.raw_parse_sents(arg2s, True)

    # arg1s = []
    # arg2s = []
    # # ind = 0
    # for line in test:
    #     # ind += 1
    #     # print ind
    #     args = line.split("||||")
    #     arg1s.append(args[0].strip().encode('utf-8'))
    #     arg2s.append(args[1].strip().encode('utf-8'))

    # print len(arg1s), len(arg2s)
    # test_parse_arg1s = parser.raw_parse_sents(arg1s, True)
    # test_parse_arg2s = parser.raw_parse_sents(arg2s, True)

    # pickle.dump( train_parse_arg1s, open( "train_parse_arg1s.p", "wb" ) )
    # pickle.dump( train_parse_arg2s, open( "train_parse_arg2s.p", "wb" ) )

    # pickle.dump( dev_parse_arg1s, open( "dev_parse_arg1s.p", "wb" ) )
    # pickle.dump( dev_parse_arg2s, open( "dev_parse_arg2s.p", "wb" ) )

    # pickle.dump( test_parse_arg1s, open( "test_parse_arg1s.p", "wb" ) )
    # pickle.dump( test_parse_arg2s, open( "test_parse_arg2s.p", "wb" ) )

        # sent1 = parser.raw_parse(args[0])
    train_parse_arg1s = pickle.load( open( "train_parse_arg1s.p", "rb" ) )
    train_parse_arg2s = pickle.load( open( "train_parse_arg2s.p", "rb" ) )
    dev_parse_arg1s = pickle.load( open( "dev_parse_arg1s.p", "rb" ) )
    dev_parse_arg2s = pickle.load( open( "dev_parse_arg2s.p", "rb" ) )
    test_parse_arg1s = pickle.load( open( "test_parse_arg1s.p", "rb" ) )
    test_parse_arg2s = pickle.load( open( "test_parse_arg2s.p", "rb" ) )

    rule_cnt = Counter()
    for tree in train_parse_arg1s:
        scnt = gen_rules(tree)
        for rule in scnt:
            rule_cnt[rule] = rule_cnt[rule] + scnt[rule]
    for tree in train_parse_arg2s:
        scnt = gen_rules(tree)
        for rule in scnt:
            rule_cnt[rule] = rule_cnt[rule] + scnt[rule]

    rule_index_dict = IndexDict()
    for rule in rule_cnt:
        rule_index_dict.get_or_add(rule)

    # bulild vectors
    train_parse_arg_vec = []
    for tree1, tree2 in zip(train_parse_arg1s,train_parse_arg2s):
        scnt1 = gen_rules(tree1)
        scnt2 = gen_rules(tree2)
        vec1 = gen_vector(scnt1,rule_index_dict)
        vec2 = gen_vector(scnt2,rule_index_dict)
        vec3 = map(lambda x:min(x), zip(vec1,vec2))
        vec = vec1 + vec2 + vec3
        vec = np.array(vec, dtype=int)
        train_parse_arg_vec.append(vec)
    train_parse_arg_vec = np.array(train_parse_arg_vec, dtype=int)

    dev_parse_arg_vec = []
    for tree1, tree2 in zip(dev_parse_arg1s,dev_parse_arg2s):
        scnt1 = gen_rules(tree1)
        scnt2 = gen_rules(tree2)
        vec1 = gen_vector(scnt1,rule_index_dict)
        vec2 = gen_vector(scnt2,rule_index_dict)
        vec3 = map(lambda x:min(x), zip(vec1,vec2))
        vec = vec1 + vec2 + vec3
        vec = np.array(vec, dtype=int)
        dev_parse_arg_vec.append(vec)
    dev_parse_arg_vec = np.array(dev_parse_arg_vec, dtype=int)

    test_parse_arg_vec = []
    for tree1, tree2 in zip(test_parse_arg1s,test_parse_arg2s):
        scnt1 = gen_rules(tree1)
        scnt2 = gen_rules(tree2)
        vec1 = gen_vector(scnt1,rule_index_dict)
        vec2 = gen_vector(scnt2,rule_index_dict)
        vec3 = map(lambda x:min(x), zip(vec1,vec2))
        vec = vec1 + vec2 + vec3
        vec = np.array(vec, dtype=int)
        test_parse_arg_vec.append(vec)
    test_parse_arg_vec = np.array(test_parse_arg_vec, dtype=int)
    print test_parse_arg_vec
    

    # print rule_set
    # print len(rule_set)

    # for word in vol.dict:
    #     print word, vol.get(word)

    f = h5py.File("pr.hdf5", "w")
    f['train_parse_arg'] = train_parse_arg_vec
    f['dev_parse_arg'] = dev_parse_arg_vec
    f['test_parse_arg'] = test_parse_arg_vec
    
    # f_train.close()
    # f_dev.close()
    # f_test.close()