#!/usr/bin/python

# This file is part of BiRNN_AutoPA - automatic extraction of pre-aspiration 
# from speech segments in audio files.
#
# Copyright (c) 2017 Yaniv Sheena


import numpy as np
import dynet as dy
# dynet for GPU:
#import _gdynet as dy 
#dy.init()
import random


NUM_OF_FEATURES_PER_FRAME = 8

class BiRNNPredictor(object):
    def __init__(self, rnn_input_dim=NUM_OF_FEATURES_PER_FRAME, rnn_output_dim=80, mlp_hid_dim=40, mlp_out_dim=3, load_from_file=''):

        # define the parameters
        self.model = dy.Model()

        # if this option is enabled - try to load the model's parameters from a file
        if load_from_file:
            self.load_params(load_from_file)
        # init the model's parameters
        else:
            # build BiLSTM
            self.birnn = dy.BiRNNBuilder(2, rnn_input_dim, rnn_output_dim, self.model, dy.LSTMBuilder)

            # first MLP layer params
            self.pW1 = self.model.add_parameters((mlp_hid_dim, rnn_output_dim))
            self.pb1 = self.model.add_parameters(mlp_hid_dim)

            # hidden MLP layer params
            self.pW2 = self.model.add_parameters((mlp_out_dim, mlp_hid_dim))
            self.pb2 = self.model.add_parameters(mlp_out_dim)

    def predict_probs(self, seq):
        '''
            Propagate through the network and output the probabilties
            of the classes of each element 
        '''         
        # Feed the input sequence into the BiRNN and get the representation of
        # its elements
        rnn_outputs = self.birnn_layer(seq)

        # Feed all the BiRNN outputs (y1..yn) into the MLP and get
        # a list of softmaxes
        return [self.do_mlp(y) for y in rnn_outputs]

    def birnn_layer(self, seq):
        ''' Feed the input sequence into the BiRNN and return the representation of
            all the elemetns in the sequence (y1,..,yn) '''

        # build expressions list for the BiRNN's input
        in_seq = []
        for vec in seq:
           expr_vec = dy.vecInput(NUM_OF_FEATURES_PER_FRAME)
           expr_vec.set(vec)
           in_seq.append(expr_vec)

        # Feed the sequence of vectors into the BiRNN
        rnn_outputs = self.birnn.transduce(in_seq)

        return rnn_outputs

    def do_mlp(self, x):
        ''' Propagate the given vector through a one hidden layer MLP '''
        h = self.mlp_layer1(x)
        y = self.mlp_layer2(h)
        return dy.softmax(y)

    def mlp_layer1(self, x):
        W = dy.parameter(self.pW1)
        b = dy.parameter(self.pb1)
        return dy.rectify(W*x+b)

    def mlp_layer2(self, x):
        W = dy.parameter(self.pW2)
        b = dy.parameter(self.pb2)
        return W*x+b

    def predict_sequence(self, seq, get_probs=False):
        ''' Get a sequence of vectors and predict a sequence of corresponding labels using
            the network (can be used for inference)'''
        dy.renew_cg()
        probs_list = self.predict_probs(seq)
        if get_probs:
            return [p.npvalue() for p in probs_list] 

        return [np.argmax(p.npvalue()) for p in probs_list]

    def get_label(self, cur_index, left_index, right_index):
        ''' Get the real label of the i'st frame, corresponding to the indexes of the event boundaries '''
        label = None
        # pre-event segment
        if cur_index < left_index:
            label = 0
        # event segment
        elif ((cur_index >= left_index) and  (cur_index <= right_index)):
            label = 1
        # post event segment
        else:
            label = 2

        return label

    def train_model(self, train_data, dev_data, learning_rate, iterations, params_file):
        ''' Train the network '''
        #trainer = dy.SimpleSGDTrainer(self.model)
        trainer = dy.AdamTrainer(self.model)
        best_dev_loss = 1e3
        best_iter = 0
        consecutive_no_improve = 0
        print 'Start training the model..'
        for ITER in xrange(iterations):
            random.shuffle(train_data)
            train_closs = 0.0
            train_success = 0
            num_of_train_frames = 0
            for seq, left_index, right_index in train_data:
                dy.renew_cg()
                probs_list = self.predict_probs(seq)
                # get the losses of the frame-based classifications
                losses = []
                for i,probs in enumerate(probs_list):
                    # get the current gold label
                    label = self.get_label(i, left_index, right_index)

                    # get negative log loss
                    curr_loss = -dy.log(dy.pick(probs, label))
                    losses.append(curr_loss)

                    # calculate precision
                    vals = probs.npvalue()
                    if np.argmax(vals) == label:
                        train_success += 1

                    num_of_train_frames += 1

                # sum all losses
                sum_loss = dy.esum(losses)

                # back propagation
                train_closs += sum_loss.value()
                sum_loss.backward()
                trainer.update(learning_rate)
                
            # check performance on dev set
            dev_closs = 0.0
            dev_success = 0
            num_of_dev_frames = 0
            for seq, left_index, right_index in dev_data:
                dy.renew_cg()
                probs_list = self.predict_probs(seq)

                # get the loss and the precision of each segment
                for i,probs in enumerate(probs_list):
                    # get the current gold label
                    label = self.get_label(i, left_index, right_index)

                    # calculate loss
                    vals = probs.npvalue()
                    dev_closs += -np.log(vals[label])

                    # calculate precision
                    if np.argmax(vals) == label:
                        dev_success += 1

                    num_of_dev_frames += 1

            avg_dev_loss = dev_closs/num_of_dev_frames

            print "Train accuracy: %s | Dev accuracy: %s | Dev avg loss: %s" % (float(train_success)/num_of_train_frames, 
                                                                                float(dev_success)/num_of_dev_frames, avg_dev_loss)

            # check if it's the best (minimum) loss so far
            if avg_dev_loss < best_dev_loss:
                best_dev_loss = avg_dev_loss
                consecutive_no_improve = 0
                # store parameters after each loss improvement
                print 'Best dev loss so far - storing parameters in %s' % params_file
                self.store_params(params_file)
            else:
                consecutive_no_improve += 1

            # after 5 consecutive epochs without improvements - stop training
            if consecutive_no_improve == 5:
                print 'No loss improvements - stop training!'
                return

        print 'Learning process has finished!'

    def store_params(self, fpath):
        self.model.save(fpath, [self.birnn, self.pW1, self.pb1, self.pW2, self.pb2])

    def load_params(self, fpath):
        (self.birnn, self.pW1, self.pb1, self.pW2, self.pb2) = self.model.load(fpath)