import cPickle
import argparse

parser = argparse.ArgumentParser(
            "Computes the coverage of a shortlist in a corpus file")
parser.add_argument("--vocab",
                    required=True, help="Vocabulary to use (.pkl)")
parser.add_argument("--text",
                    required=True, help="Beam size, turns on beam-search")
args = parser.parse_args()

with open(args.vocab, 'rb') as f:
    d = cPickle.load(f)

with open(args.text, 'rb') as f:
    text = f.read().splitlines()


n_words = 0
n_unks = 0
split_vocab = 0
split_vocabulary = {}

for line in text:
    for word in line.split():
        if split_vocabulary.get(word) is None:
            split_vocabulary[word] = split_vocab
            split_vocab += 1
            if d.get(word) is None:
                n_unks += 1
        n_words += 1

print "Coverage: %f (%d unknown words out of %d of a total of %d)"%((float)(split_vocab - n_unks)/split_vocab, n_unks, split_vocab, n_words)
