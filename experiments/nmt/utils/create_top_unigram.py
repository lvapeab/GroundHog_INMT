import numpy
import cPickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--top-unigram", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

with open(args.top_unigram,'rb') as f:
    top_unigram = cPickle.load(f)

new_dict = {}
for old_key in top_unigram:
    if old_key == '<eps>': # Don't consider the empty string
        continue
    new_dict[old_key] = top_unigram[old_key][0]

with open(args.output,'wb') as f:
    cPickle.dump(new_dict, f, -1)
