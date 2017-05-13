import numpy as np
import dynet as dy
#import _gdynet as dy # dynet for GPU
#dy.init()
import random


NUM_OF_FEATURES_PER_FRAME = 8

class BiRNNBinaryPredictor(object):
    def __init__(self, rnn_input_dim=NUM_OF_FEATURES_PER_FRAME, rnn_output_dim=30, mlp_hid_dim=20, mlp_out_dim=2, load_from_file=''):

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
            self.pW1 = self.model.add_parameters((mlp_hid_dim, rnn_output_dim), init=dy.GlorotInitializer())
            self.pb1 = self.model.add_parameters(mlp_hid_dim, init=dy.GlorotInitializer())

            # hidden MLP layer params
            self.pW2 = self.model.add_parameters((mlp_out_dim, mlp_hid_dim), init=dy.GlorotInitializer())
            self.pb2 = self.model.add_parameters(mlp_out_dim, init=dy.GlorotInitializer())

    def predict_probs(self, seq, index=None, train_mode=False):
        '''
            Propagate through the network and output the probabilties
            of the #index element in the given sequence (in case index is given),
            or all the elements' probabilties (in the default case - index=None)
        '''         
        # Feed the input sequence into the BiRNN and get the representation of
        # its elements
        rnn_outputs = self.birnn_layer(seq)

        # predict all the sequence at once - used in inference
        if index is None:
            # Feed all the BiRNN outputs (y1..yn) into the MLP
            return [self.do_mlp(y) for y in rnn_outputs]

        # predict a specific index
        else:
            # get Yi
            rnn_y_i = rnn_outputs[index]
            # Dropout layer - only when training
            #if train_mode:
            #    dy.dropout(rnn_y_i, 0.3)

            # Feed the BiRNN representation (Yindex) into the MLP
            return self.do_mlp(rnn_y_i)

    def birnn_layer(self, seq):
        ''' Feed the input sequence into the BiRNN and return the representation of
            all the elemetns in the sequence (y) '''

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

    def do_loss(self, probs, label):
        return -dy.log(dy.pick(probs, label))

    def classify(self, seq, index, label):
        ''' Use the network to predict the label with the highest probability, return
            also the negative log loss of the given label '''
        dy.renew_cg()
        probs = self.predict_probs(seq, index)
        vals = probs.npvalue()
        return np.argmax(vals), -np.log(vals[label])

    def predict_sequence(self, seq):
        ''' Get a sequence of vectors and predict a sequence of corresponding binary values using
            the network'''
        dy.renew_cg()
        probs_list = self.predict_probs(seq)
        return [np.argmax(p.npvalue()) for p in probs_list]

    def train_model(self, train_data, dev_data, learning_rate, iterations):
        ''' Train the network '''
        #trainer = dy.SimpleSGDTrainer(self.model)
        trainer = dy.AdamTrainer(self.model)
        best_dev_loss = 1e3
        best_iter = 0
        print 'Start training the model..'
        for ITER in xrange(iterations):
            random.shuffle(train_data)
            closs = 0.0
            train_success = 0
            for seq, index, label in train_data:
                dy.renew_cg()
                probs = self.predict_probs(seq, index, train_mode=True)
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

    def store_params(self, fpath):
        self.model.save(fpath, [self.birnn, self.pW1, self.pb1, self.pW2, self.pb2])

    def load_params(self, fpath):
        (self.birnn, self.pW1, self.pb1, self.pW2, self.pb2) = self.model.load(fpath)