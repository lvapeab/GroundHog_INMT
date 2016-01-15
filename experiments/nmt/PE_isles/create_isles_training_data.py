import argparse
from experiments.nmt import prototype_phrase_state
import logging
import numpy
import cPickle

from isles_utils import find_isles

logger = logging.getLogger(__name__)

def correct_isles(hyp_file, ref_file, model_path, w2i, output_prefix, unk_id=0, func='sum'):

    embedding = numpy.load(model_path)['W_0_dec_approx_embdr']
    bias = numpy.load(model_path)['b_0_dec_approx_embdr']
    emb = embedding + bias

    del embedding
    del bias
    emb_size = emb.shape[1]
    fhyp = open(hyp_file, 'r')
    fref = open(ref_file, 'r')
    target_lines = fref.read().split('\n')
    if target_lines[-1] == '':
        target_lines = target_lines[:-1]
    train_data = numpy.zeros((0, emb_size), dtype="float32")
    expected_words = []
    for n_line, line in enumerate(fhyp):
        if n_line % 100 == 0:
            logger.info("Processed " + str(n_line) + " lines")
        target_line = target_lines[n_line].strip().split()
        isles = find_isles(line.strip().split(), target_line)
        if len(isles[1]) > 0:
            for ref_elem in range(len(isles[1])-1):
                word_repr = numpy.zeros(emb_size, dtype="float32")
                for w in isles[1][ref_elem][1]:
                    index_word = w2i[w] if w2i.get(w) is not None else unk_id
                    word_repr = numpy.add(word_repr, emb[index_word])
                if func == 'avg':
                    word_repr /= len(isles[1][ref_elem][1])
                train_data = numpy.append(train_data, [word_repr], axis=0)
                expected_words.append(isles[1][ref_elem+1][0] - isles[1][ref_elem][0] + len(isles[1][ref_elem][1]) -1)
            ref_elem = len(isles[1])-1
            word_repr = numpy.zeros(emb_size, dtype="float32")
            for w in isles[1][ref_elem][1]:
                    index_word = w2i[w] if w2i.get(w) is not None else unk_id
                    word_repr = numpy.add(word_repr, emb[index_word])
            if func == 'avg':
                word_repr /= len(isles[1][ref_elem][1])
            train_data = numpy.append(train_data, [word_repr], axis=0)
            expected_words.append(len(target_line) - isles[1][ref_elem][0] - len(isles[1][ref_elem][1]) + len(isles[1][ref_elem][1]))
        # else:
        #     for w in range(0, len(target_line)):
        #         word_repr = emb[w2i[w]] if w2i.get(w) is not None else emb[unk_id]
        #         train_data = numpy.append(train_data, [word_repr], axis=0)
        #         expected_words.append(len(target_line) - w)
    classes = numpy.asarray(expected_words, dtype="int64")
    numpy.savez(output_prefix + '.npz', embeddings=train_data, classes=classes)


    logger.debug("End")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, help="State to use")
    # Paths
    parser.add_argument("--hyp", help="System hypotheses")
    parser.add_argument("--ref", help="References")
    parser.add_argument("--out", help="Prefix for output files")
    parser.add_argument("--func",  default="sum", help="Function for combining continuous representations of words. "
                                                       "[sum | avg]")

    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print more stuff")
    # Additional arguments
    parser.add_argument("changes",  nargs="?", help="Changes to state", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    state = prototype_phrase_state()
    with open(args.state) as src:
        state.update(cPickle.load(src))
    logging.basicConfig(level=getattr(logging, state['level']),
                        format=" %(asctime)s: %(name)s: %(levelname)s: %(message)s")
    if args.verbose:
        logger.setLevel(level=logging.DEBUG)
        logger.debug("I'm being verbose!")
    else:
        logger.setLevel(level=logging.INFO)

    hyp_file = args.hyp
    ref_file = args.ref
    indx_word_trg = cPickle.load(open(state['word_indx_trgt'], 'rb'))
    unk_id = state['unk_sym_target']
    correct_isles(hyp_file, ref_file, args.model_path, indx_word_trg,
                  output_prefix=args.out, unk_id=unk_id,  func=args.func)

if __name__ == "__main__":
    main()
