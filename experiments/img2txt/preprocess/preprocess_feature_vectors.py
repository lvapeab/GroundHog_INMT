#!/usr/bin/env python

import argparse
import cPickle
import gzip
import bz2
import logging
import os
import sys
import numpy
import tables

import csv
import ast

from collections import Counter
from operator import add
from numpy.lib.stride_tricks import as_strided

parser = argparse.ArgumentParser(
    description="""
This takes a list of .txt or .txt.gz files and does word counting and
creating a dictionary (potentially limited by size). It uses this
dictionary to binarize the text into a numeric format (replacing OOV
words with 1) and create n-grams of a fixed size (padding the sentence
with 0 for EOS and BOS markers as necessary). The n-gram data can be
split up in a training and validation set.

The n-grams are saved to HDF5 format whereas the dictionary, word counts
and binarized text are all pickled Python objects.
""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("input", type=argparse.FileType('r'), nargs="+",
                    help="The input files")

parser.add_argument("-o", "--overwrite", action="store_true",
                    help="overwrite earlier created files, also forces the "
                         "program not to reuse count files")

parser.add_argument("-b", "--binarized-vectors", default='binarized_vectors.pkl',
                    help="the name of the pickled binarized vectors file")

parser.add_argument("-d", "--dimension", type=int, default=2048,
                    help="dimension of the feature vectors")

parser.add_argument("-c", "--csv-input", type=bool, default=True,
                    help="process the feature vectors from a csv file")

def open_files():
    base_filenames = []
    for i, input_file in enumerate(args.input):
        dirname, filename = os.path.split(input_file.name)
        if filename.split(os.extsep)[-1] == 'gz':
            base_filename = filename.rstrip('.gz')
        elif filename.split(os.extsep)[-1] == 'bz2':
            base_filename = filename.rstrip('.bz2')
        else:
            base_filename = filename
        if base_filename.split(os.extsep)[-1] == 'txt':
            base_filename = base_filename.rstrip('.txt')
        if filename.split(os.extsep)[-1] == 'gz':
            args.input[i] = gzip.GzipFile(input_file.name, input_file.mode,
                                          9, input_file)
        elif filename.split(os.extsep)[-1] == 'bz2':
            args.input[i] = bz2.BZ2File(input_file.name, input_file.mode)
        base_filenames.append(base_filename)
    return base_filenames


def safe_pickle(obj, filename):
    if os.path.isfile(filename) and not args.overwrite:
        logger.warning("Not saving %s, already exists." % (filename))
    else:
        if os.path.isfile(filename):
            logger.info("Overwriting %s." % filename)
        else:
            logger.info("Saving to %s." % filename)
        with open(filename, 'wb') as f:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)


def safe_hdf(array, name):
    if os.path.isfile(name + '.hdf') and not args.overwrite:
        logger.warning("Not saving %s, already exists." % (name + '.hdf'))
    else:
        if os.path.isfile(name + '.hdf'):
            logger.info("Overwriting %s." % (name + '.hdf'))
        else:
            logger.info("Saving to %s." % (name + '.hdf'))
        with tables.openFile(name + '.hdf', 'w') as f:
            atom = tables.Atom.from_dtype(array.dtype)
            filters = tables.Filters(complib='blosc', complevel=5)
            ds = f.createCArray(f.root, name.replace('.', ''), atom,
                                array.shape, filters=filters)
            ds[:] = array



def binarize_csv():
    ### VIDEOS!!
    binarized_corpora = []
    total_vectors_count = 0
    total_videos_count = 0
    for input_file, base_filename in zip(args.input, base_filenames):
        input_filename = os.path.basename(input_file.name)
        logger.info("Reading and binarizing csv file: %s." % (input_filename))

        with open(input_file.name, 'rb') as csvfile:
                # fieldnames = ['ID', 'start_second', 'end_second', 'descriptions_list']
                csv_reader = csv.reader(csvfile, delimiter=",")
                video_frames = []
                for video in csv_reader :
                    if video == ['E'] :
                        if total_videos_count % 100 == 0:
                            print total_videos_count,
                            sys.stdout.flush()
                        elif total_videos_count % 10 == 0:
                            print '.',
                            sys.stdout.flush()

                        binarized_corpora.append(video_frames)
                        video_frames = []
                        total_videos_count += 1
                    else :
                        video_img = map(float, video)
                        if video_img != [''] :
                            assert len(video_img) % args.dimension == 0
                            video_frames.append(video_img)
                            total_vectors_count += 1



    # endfor input_file in args.inp
    safe_pickle(binarized_corpora, args.binarized_vectors)
    logger.info('Binarized ' +  str (total_videos_count) + ' videos (' + str (total_vectors_count) + ') vectors into file ' + args.binarized_vectors)





def binarize():

    binarized_corpora = []

    total_vectors_count = 0
    for input_file, base_filename in \
            zip(args.input, base_filenames):
        input_filename = os.path.basename(input_file.name)
        logger.info("Binarizing %s." % (input_filename))
        binarized_corpus = []

        for sentence_count, sentence in enumerate(input_file):
            vectors = sentence.strip().split(' ')
            if vectors != [''] :
                assert len(vectors) % args.dimension == 0
                binarized_corpus.append(map(float, vectors))
                total_vectors_count += 1

        binarized_corpora += binarized_corpus
    # endfor input_file in args.input
    safe_pickle(binarized_corpora, args.binarized_vectors)
    logger.info('Binarized ' + str (total_vectors_count) + ' vectors into file ' + args.binarized_vectors)





if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('preprocess')
    args = parser.parse_args()
    base_filenames = open_files()
    if args.csv_input :
        binarize_csv()
    else :
        binarize()
