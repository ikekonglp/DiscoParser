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
import StanfordDependencies

MAX_LENGTH = 100
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

    def get(self, word, level=0):
        if word in self.dict:
            return self.dict[word] + (level * (self.ind-1))
        else:
            return 2 + (level * (self.ind-1))

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


def parse_tokenized_sentences(parser, sentences, verbose=False):
    cmd = [
        'edu.stanford.nlp.parser.lexparser.LexicalizedParser',
        '-model', parser.model_path,
        '-sentences', 'newline',
        '-outputFormat', 'penn',
        '-tokenized',
    ]
    return parser._parse_trees_output(parser._execute(cmd, '\n'.join(sentences), verbose))


if __name__ == '__main__':
    f_train = open("train_imp", "r")
    f_dev = open("dev_imp", "r")
    f_test = open("test_imp", "r")

    train = f_train.readlines()
    dev = f_dev.readlines()
    test = f_test.readlines()
    
    # cnt = Counter()
    # relations = IndexDict()
    # vol = IndexDict()

    os.environ['STANFORD_PARSER'] = STANFORD_DIR + "stanford-parser.jar"
    os.environ['STANFORD_MODELS'] = STANFORD_DIR + "stanford-parser-3.3.0-models.jar"

    parser = stanford.StanfordParser(model_path= STANFORD_DIR + "englishPCFG.ser.gz")
    # st = (" ".join(nltk.word_tokenize("Hello, My (name) is Melroy."))).replace("(", "-LRB-").replace(")","-RRB-")
    # sentences = parse_tokenized_sentences(parser, [st])
    # print st
    # print sentences

    # sd = StanfordDependencies.get_instance(version='3.3.0')

    # print [sd.convert_tree(s._pprint_flat(nodesep='', parens='()', quotes=False)) for s in sentences]


    arg1s = []
    arg2s = []
    ind = 0
    for line in train:
        ind += 1
        print ind
        args = line.split("||||")
        arg1s.append(((" ".join( [w for w in nltk.word_tokenize(args[0].strip()) if len(w) > 0] )).strip()).replace("(", "-LRB-").replace(")","-RRB-").encode('utf-8') )
        arg2s.append(((" ".join( [w for w in nltk.word_tokenize(args[1].strip()) if len(w) > 0] )).strip()).replace("(", "-LRB-").replace(")","-RRB-").encode('utf-8') )

    print len(arg1s), len(arg2s)
    train_parse_arg1s = parse_tokenized_sentences(parser, arg1s, True)
    train_parse_arg2s = parse_tokenized_sentences(parser, arg2s, True)

    arg1s = []
    arg2s = []
    ind = 0
    for line in dev:
        ind += 1
        print ind
        args = line.split("||||")
        arg1s.append(((" ".join( [w for w in nltk.word_tokenize(args[0].strip()) if len(w) > 0] )).strip()).replace("(", "-LRB-").replace(")","-RRB-").encode('utf-8') )
        arg2s.append(((" ".join( [w for w in nltk.word_tokenize(args[1].strip()) if len(w) > 0] )).strip()).replace("(", "-LRB-").replace(")","-RRB-").encode('utf-8') )

    print len(arg1s), len(arg2s)
    dev_parse_arg1s = parse_tokenized_sentences(parser, arg1s, True)
    dev_parse_arg2s = parse_tokenized_sentences(parser, arg2s, True)

    arg1s = []
    arg2s = []
    ind = 0
    for line in test:
        ind += 1
        print ind
        args = line.split("||||")
        arg1s.append(((" ".join( [w for w in nltk.word_tokenize(args[0].strip()) if len(w) > 0] )).strip()).replace("(", "-LRB-").replace(")","-RRB-").encode('utf-8') )
        arg2s.append(((" ".join( [w for w in nltk.word_tokenize(args[1].strip()) if len(w) > 0] )).strip()).replace("(", "-LRB-").replace(")","-RRB-").encode('utf-8') )

    print len(arg1s), len(arg2s)

    test_parse_arg1s = parse_tokenized_sentences(parser, arg1s, True)
    test_parse_arg2s = parse_tokenized_sentences(parser, arg2s, True)

    pickle.dump( train_parse_arg1s, open( "train_parse_arg1s.p", "wb" ) )
    pickle.dump( train_parse_arg2s, open( "train_parse_arg2s.p", "wb" ) )

    pickle.dump( dev_parse_arg1s, open( "dev_parse_arg1s.p", "wb" ) )
    pickle.dump( dev_parse_arg2s, open( "dev_parse_arg2s.p", "wb" ) )

    pickle.dump( test_parse_arg1s, open( "test_parse_arg1s.p", "wb" ) )
    pickle.dump( test_parse_arg2s, open( "test_parse_arg2s.p", "wb" ) )


    # train_parse_arg1s = pickle.load( open( "train_parse_arg1s.p", "rb" ) )
    # train_parse_arg2s = pickle.load( open( "train_parse_arg2s.p", "rb" ) )
    # dev_parse_arg1s = pickle.load( open( "dev_parse_arg1s.p", "rb" ) )
    # dev_parse_arg2s = pickle.load( open( "dev_parse_arg2s.p", "rb" ) )
    # test_parse_arg1s = pickle.load( open( "test_parse_arg1s.p", "rb" ) )
    # test_parse_arg2s = pickle.load( open( "test_parse_arg2s.p", "rb" ) )
