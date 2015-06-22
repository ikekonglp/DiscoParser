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
from nltk import Tree
import pickle

MAX_LENGTH = 100

# The index dictionary, 1 is for padding, 2 is OOV
# The level gives you different indexing system when you want to sepearte the feature indices
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

    def get(self, word, level=0):
        if word in self.dict:
            return self.dict[word] + (level * (self.ind-1))
        else:
            return 2 + (level * (self.ind-1))

    def length(self):
        return (self.ind-1)

def gen_word_vector(tokens_arg, vol):
    num_arg = [vol.get(word) for word in tokens_arg]

    assert(len(num_arg) > 0)

    arg = []
    front = 0
    for num in num_arg:
        arg.append(num)

    while len(arg) < MAX_LENGTH:
        if front < 5:
            arg.insert(0, 1)
            front += 1
        else:
            arg.append(1)
    return arg

def gen_parse_feature_vector(tree1, tree2):
    train_parse_arg_vec = []

    scnt1 = gen_rules(tree1)
    scnt2 = gen_rules(tree2)
    common_rules = [r for r in scnt1 if (r in scnt2)]
    for r in scnt1:
        train_parse_arg_vec.append(rule_index_dict.get(r, 0))
    for r in scnt2:
        train_parse_arg_vec.append(rule_index_dict.get(r, 1))
    for r in common_rules:
        train_parse_arg_vec.append(rule_index_dict.get(r, 2))

    while len(train_parse_arg_vec) < (3 * MAX_LENGTH):
        train_parse_arg_vec.append(1)
    return train_parse_arg_vec

def gen_representation(dataset, vol, relations, parse_arg1s, parse_arg2s, rule_index_dict, throw_away=False):
    targ1 = []
    targ2 = []
    trelation = []
    tparse = []
    data_ind = -1
    for line in dataset:
        data_ind += 1
        tokens_arg1, tokens_arg2, relation = extract_fields(line)
        arg1 = gen_word_vector(tokens_arg1, vol)
        arg2 = gen_word_vector(tokens_arg2, vol)

        if (throw_away) and (len(arg2) > MAX_LENGTH or len(arg1) > MAX_LENGTH):
            continue

        train_parse_arg_vec = gen_parse_feature_vector(parse_arg1s[data_ind], parse_arg2s[data_ind])

        targ1.append(arg1)
        targ2.append(arg2)
        tparse.append(train_parse_arg_vec)
        trelation.append(relations.get(relation))

        assert(len(arg1) == MAX_LENGTH)
        assert(len(arg2) == MAX_LENGTH)
        assert(len(train_parse_arg_vec) == (3 * MAX_LENGTH))
        assert(len(targ1) == len(trelation))
        
    return [targ1, targ2, trelation, tparse]

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

def load_parses():
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
    return (rule_index_dict, train_parse_arg1s, train_parse_arg2s, dev_parse_arg1s, dev_parse_arg2s, test_parse_arg1s, test_parse_arg2s)

def read_in_data():
    f_train = open("train_imp", "r")
    f_dev = open("dev_imp", "r")
    f_test = open("test_imp", "r")

    train = f_train.readlines()
    dev = f_dev.readlines()
    test = f_test.readlines()

    f_train.close()
    f_dev.close()
    f_test.close()

    return (train, dev, test)

def load_dictionary(train):
    cnt = Counter()
    relations = IndexDict()
    vol = IndexDict()

    for line in train:
        tokens_arg1, tokens_arg2, relation = extract_fields(line)
        for w in tokens_arg1:
            cnt[w] += 1
        for w in tokens_arg2:
            cnt[w] += 1
        relations.get_or_add(relation)

    for w in cnt:
        if cnt[w] > 1:
            vol.get_or_add(w)
    return (cnt, relations, vol)

def extract_fields(line):
    args = line.split("||||")

    tokens_arg1 = [word.strip() for word in nltk.word_tokenize(args[0])]
    tokens_arg2 = [word.strip() for word in nltk.word_tokenize(args[1])]
    relation = args[2].strip()

    return (tokens_arg1, tokens_arg2, relation)

if __name__ == '__main__':
    train, dev, test = read_in_data()

    cnt, relations, vol = load_dictionary(train)

    # print relation table
    f_relation_table = open("relation_table", "w")
    for relation in relations.dict:
        f_relation_table.write(relation + "\t" + str(relations.get(relation)) + "\n")
    f_relation_table.close()
    ####

    verts = load_word2vec(vol)

    rule_index_dict, train_parse_arg1s, train_parse_arg2s, dev_parse_arg1s, dev_parse_arg2s, test_parse_arg1s, test_parse_arg2s = load_parses()

    f = h5py.File("mr.hdf5", "w")
    f['train_arg1'], f['train_arg2'], f['train_label'], f['train_parse'] = ([np.array(ele, dtype=int) for ele in gen_representation(train, vol, relations, train_parse_arg1s, train_parse_arg2s, rule_index_dict, True)])
    f['dev_arg1'], f['dev_arg2'], f['dev_label'], f['dev_parse'] = ([np.array(ele, dtype=int) for ele in gen_representation(dev, vol, relations, dev_parse_arg1s, dev_parse_arg2s, rule_index_dict)])
    f['test_arg1'], f['test_arg2'], f['test_label'], f['test_parse'] = ([np.array(ele, dtype=int) for ele in gen_representation(test, vol, relations, test_parse_arg1s, test_parse_arg2s, rule_index_dict)])
    f['embeding'] = np.array(verts)
    
    print np.array(verts).shape
