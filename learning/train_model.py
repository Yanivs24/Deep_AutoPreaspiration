#!/usr/bin/python

import numpy as np
import dynet as dy
import argparse
import random

from model.model import BiRNNBinaryPredictor

AUTHOR={'Yaniv Sheena'}


NUM_OF_FEATURES_PER_FRAME = 8
DEV_SET_PROPORTION        = 0.2


def read_examples(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    return [l.strip().split(',') for l in lines]

if __name__ == '__main__':

      # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", help="A path to the training set")
    parser.add_argument("params_path", help="A path to a file in which the trained model parameters will be stored")
    parser.add_argument('--num_iters', help='Number of iterations (epochs)', default=5, type=int)
    parser.add_argument('--learning_rate', help='The learning rate', default=0.1, type=float)
    args = parser.parse_args()

    raw_dataset = read_examples(args.train_path)
    print 'Got %s training examples!' % (len(raw_dataset))

    # build dataset according to the trainer expected format
    dataset = []
    for ex in raw_dataset:
        # Get flat vector 
        flat_seq = np.fromstring(ex[0], sep=' ')

        # each NUM_OF_FEATURES_PER_FRAME values is a frame
        num_of_frames = float(len(flat_seq)) / NUM_OF_FEATURES_PER_FRAME
        # handle error
        if not num_of_frames.is_integer():
            raise ValueError("Input vector size is not a multiple of the features amount (%s)" % NUM_OF_FEATURES_PER_FRAME)

        # cast to integer
        num_of_frames = int(num_of_frames)

        # Reshape it to the original size - 2d list of vectors of dim 'NUM_OF_FEATURES_PER_FRAME' -
        # this represents the speech segment
        speech_seq = [flat_seq[NUM_OF_FEATURES_PER_FRAME*i: NUM_OF_FEATURES_PER_FRAME*(i+1)] for i in range(num_of_frames)]

        # append to the dataset
        dataset.append((speech_seq, int(ex[1]), int(ex[2])))

    # split the dataset into training set and validation set
    train_set_size = int((1-DEV_SET_PROPORTION) * len(dataset))
    train_data = dataset[:train_set_size]
    dev_data   = dataset[train_set_size:]

    # build a new model 
    my_model = BiRNNBinaryPredictor()

    # train the model
    my_model.train_model(train_data, dev_data,  args.learning_rate, args.num_iters)

    # store parameters in a file
    print "Storing model parameters in %s" % args.params_path
    my_model.store_params(args.params_path)
