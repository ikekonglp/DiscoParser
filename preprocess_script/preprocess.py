import argparse
import codecs
import os
import re
import sys

parser = argparse.ArgumentParser(description='')
parser.add_argument('inputf', type=str, metavar='', help='')

A = parser.parse_args()

pattern_split = re.compile(r'^_+$')
pattern_type = re.compile(r'^_+.+_+$')

pattern_explicit = re.compile(r'^_+Explicit_+$')
pattern_implicit = re.compile(r'^_+Implicit_+$')
pattern_entrel = re.compile(r'^_+EntRel_+$')
pattern_norel = re.compile(r'^_+NoRel_+$')
pattern_altlex = re.compile(r'^_+AltLex_+$')

pattern_arg1 = re.compile(r'^_+Arg1_+$')
pattern_arg2 = re.compile(r'^_+Arg2_+$')

pattern_sup1 = re.compile(r'^_+Sup1_+$')
pattern_sup2 = re.compile(r'^_+Sup2_+$')

pattern_text = re.compile(r'^#+ Text #+$')

pattern_dd = re.compile(r'^\d\d$')
pattern_filename = re.compile(r'^wsj.+$')

pattern_empty_line = re.compile(r'^#+$')

def process_file(f):
    f = open(f, "r")
    # match explicit
    lines = [l.strip() for l in f]
    i = 0
    store_info = []
    while i < len(lines):
        line = lines[i]
        i += 1
        if pattern_split.match(line):
            if len(store_info) > 1:
                process_unit(store_info)
            store_info = []
        else:
            store_info.append(line)

        
    f.close()

def find_arg12(store_info):
    ind_arg1 = find_first_start_at(0, pattern_arg1, store_info)
    ind_text1 = find_first_start_at(ind_arg1, pattern_text, store_info)
    text1 = store_info[ind_text1+1].strip()

    ind_arg2 = find_first_start_at(0, pattern_arg2, store_info)
    ind_text2 = find_first_start_at(ind_arg2, pattern_text, store_info)
    text2 = store_info[ind_text2+1].strip()

    return [text1, text2]


def process_unit(store_info):
    if pattern_type.match(store_info[0]):
        relation = ""
        info_type = store_info[0]
        if pattern_explicit.match(info_type):
            ind = find_first_start_at(0, pattern_sup1, store_info) if find_first_start_at(0, pattern_sup1, store_info) > 0 else find_first_start_at(0, pattern_arg1, store_info)
            relation = store_info[ind-1][(store_info[ind-1].rfind(',') + 1):].strip().split('.')
            if len(relation) > 2:
                relation = relation[:2]
            relation = ".".join(relation)

        elif pattern_implicit.match(info_type):
            ind = find_first_start_at(0, pattern_sup1, store_info) if find_first_start_at(0, pattern_sup1, store_info) > 0 else find_first_start_at(0, pattern_arg1, store_info)
            relation = store_info[ind-1][(store_info[ind-1].rfind(',') + 1):].strip().split('.')
            if len(relation) > 2:
                relation = relation[:2]
            relation =  ".".join(relation)
        elif pattern_altlex.match(info_type):
            ind = find_first_start_at(0, pattern_sup1, store_info) if find_first_start_at(0, pattern_sup1, store_info) > 0 else find_first_start_at(0, pattern_arg1, store_info)
            relation = store_info[ind-1][(store_info[ind-1].rfind(',') + 1):].strip().split('.')
            if len(relation) > 2:
                relation = relation[:2]
            relation =  ".".join(relation)
        elif pattern_entrel.match(info_type):
            relation = 'EntRel'
        elif pattern_norel.match(info_type):
            relation = 'NoRel'
        finlist = find_arg12(store_info)
        finlist.append(relation)
        print "||||".join(finlist)

def find_first_start_at(start, pattern, store_info):
    for i in xrange(start, len(store_info)):
        if pattern.match(store_info[i]):
            return i
    return -1


if __name__ == '__main__':
    for lists in os.listdir(A.inputf): 
        path = os.path.join(A.inputf, lists)
        if pattern_dd.match(lists):
            for files in os.listdir(path):
                file_path = os.path.join(path, files)
                if pattern_filename.match(files):
                    process_file(file_path)
