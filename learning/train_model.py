#!/usr/bin/python

import numpy as np
import dynet as dy
import argparse
import random

AUTHOR={'Yaniv Sheena'}


NUM_OF_FEATURES_PER_FRAME = 8
DEV_SET_PROPORTION        = 0.2


class DynetBiRNNModel(object):
    def __init__(self, rnn_input_dim=NUM_OF_FEATURES_PER_FRAME, rnn_output_dim=30, mlp_hid_dim=20, mlp_out_dim=2):

        # define the parameters
        self.model = dy.Model()

        # build BiLSTM
        self.birnn = dy.BiRNNBuilder(1, rnn_input_dim, rnn_output_dim, self.model, dy.LSTMBuilder)

        # first MLP layer params
        self.pW1 = self.model.add_parameters((mlp_hid_dim, rnn_output_dim))
        self.pb1 = self.model.add_parameters(mlp_hid_dim)

        # hidden MLP layer params
        self.pW2 = self.model.add_parameters((mlp_out_dim, mlp_hid_dim))
        self.pb2 = self.model.add_parameters(mlp_out_dim)

    def predict_probs(self, seq, index):
        '''
            Propagate through the network and output the probabilties
            of the #index element in the given sequence 
        '''         
        # Feed the input sequence into the BiRNN and get the representation of
        # the #index element (yi)
        x = self.birnn_layer(seq, index)
        # Feed the BiRNN representation into the MLP
        h = self.mlp_layer1(x)
        y = self.mlp_layer2(h)
        return dy.softmax(y)

    def birnn_layer(self, seq, index):
        ''' Feed the input sequence into the BiRNN and return the representation of
            the #index element (yi) '''

        # build expressions list for the BiRNN's input
        in_seq = []
        for vec in seq:
           expr_vec = dy.vecInput(NUM_OF_FEATURES_PER_FRAME)
           expr_vec.set(vec)
           in_seq.append(expr_vec)

        # Feed the sequence of vectors into the BiRNN
        rnn_outputs = self.birnn.transduce(in_seq)

        # output only Yindex 
        return rnn_outputs[index]

    def mlp_layer1(self, x):
        W = dy.parameter(self.pW1)
        b = dy.parameter(self.pb1)
        return dy.tanh(W*x+b)

    def mlp_layer2(self, x):
        W = dy.parameter(self.pW2)
        b = dy.parameter(self.pb2)
        return W*x+b

    def do_loss(self, probs, label):
        return -dy.log(dy.pick(probs, label))

    def classify(self, seq, index, label):
        ''' Use the network to predict the label with the highest probability, return
            also the negative log loss of the given label '''
        dy.renew_cg()
        probs = self.predict_probs(seq, index)
        vals = probs.npvalue()
        return np.argmax(vals), -np.log(vals[label])

    def predict(self, seq, index):
        ''' wrapper method - classify without returning the loss '''
        prediction, _ = self.classify(seq, index, 0)
        return prediction

    def train_model(self, train_data, dev_data, learning_rate=1, max_iterations=20):
        ''' Train the network '''
        #trainer = dy.SimpleSGDTrainer(self.model)
        trainer = dy.AdamTrainer(self.model)
        best_dev_loss = 1e3
        best_iter = 0
        print 'Start training the model..'
        for ITER in xrange(max_iterations):
            random.shuffle(train_data)
            closs = 0.0
            train_success = 0
            for seq, index, label in train_data:
                dy.renew_cg()
                probs = self.predict_probs(seq, index)
                loss = self.do_loss(probs, label)
                closs += loss.value()
                loss.backward()
                trainer.update(learning_rate)

                vals = probs.npvalue()
                if np.argmax(vals) == label:
                    train_success += 1
                
            # check performance on dev set
            success_count = 0
            dev_closs = 0.0
            for seq, index, label in dev_data:
                prediction, dev_loss = self.classify(seq, index, label)
                # accumulate loss
                dev_closs += dev_loss
                success_count += (prediction == label)

            avg_dev_loss = dev_closs/len(dev_data)
            dev_accuracy = float(success_count)/len(dev_data)

            print "Train accuracy: %s | Dev accuracy: %s | Dev avg loss: %s" % (float(train_success)/len(train_data), 
                dev_accuracy, avg_dev_loss)

        print 'Learning process has finished!'


def read_examples(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    return [l.strip().split(',') for l in lines]

if __name__ == '__main__':

      # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", help="A path to the training set")
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

    # split to dataset into training set and validation set
    train_set_size = int((1-DEV_SET_PROPORTION) * len(dataset))
    train_data = dataset[:train_set_size]
    dev_data   = dataset[train_set_size:]

    # build a new model 
    my_model = DynetBiRNNModel()

    # train the model
    my_model.train_model(train_data, dev_data)
