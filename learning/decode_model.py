import numpy as np
import dynet as dy
import argparse

from model.model import BiRNNBinaryPredictor

LEFT_WINDOW_SIZE    = 50
RIGHT_WINDOW_SIZE   = 60
FIRST_FEATURE_INDEX = 1
LAST_FEATURE_INDEX  = 9
NUM_OF_FEATURES_PER_FRAME = LAST_FEATURE_INDEX-FIRST_FEATURE_INDEX


class DynetBiRNNModel(object):
    def __init__(self, rnn_input_dim=NUM_OF_FEATURES_PER_FRAME, rnn_output_dim=30, mlp_hid_dim=20, mlp_out_dim=2):

        # define the parameters
        self.model = dy.Model()

        # # build BiLSTM
        # self.birnn = dy.BiRNNBuilder(2, rnn_input_dim, rnn_output_dim, self.model, dy.LSTMBuilder)

        # # first MLP layer params
        # self.pW1 = self.model.add_parameters((mlp_hid_dim, rnn_output_dim))
        # self.pb1 = self.model.add_parameters(mlp_hid_dim)

        # # hidden MLP layer params
        # self.pW2 = self.model.add_parameters((mlp_out_dim, mlp_hid_dim))
        # self.pb2 = self.model.add_parameters(mlp_out_dim)

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

        # output only representation of the given index 
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
        ''' wrapper of classify - classify without returning the loss '''
        prediction, _ = self.classify(seq, index, 0)
        return prediction

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

    def store_params(self, fpath):
        self.model.save(fpath, [self.birnn, self.pW1, self.pb1, self.pW2, self.pb2])

    def load_params(self, fpath):
        (self.birnn, self.pW1, self.pb1, self.pW2, self.pb2) = self.model.load(fpath)


def get_feature_files(feature_names_file):
    with open(feature_names_file) as f:
        file_names = f.readlines()

    return [line.strip() for line in file_names]

def read_features(file_name):
    return np.loadtxt(file_name, skiprows=1)[:, FIRST_FEATURE_INDEX:LAST_FEATURE_INDEX]

def smooth_binary_vector(vec):
    ''' Smooth binary vector using convolotion with 5 valued vectores of ones:
        [1,1,1,1,1], and binary threshold of 3'''
    smooth_vec = np.zeros(len(vec))
    for i in range(2, len(vec)-2):
        if sum(vec[i-2:i+3]) >= 3:
            smooth_vec[i] = 1

    return smooth_vec

def find_longest_event(vec):
    ''' finds the longest '1' subvector within a binary vector, returns the 
        indexes of the boundaries '''

    count = 0
    max_count = 0
    end_index = 0
    for i, val in enumerate(vec):
        if val == 1:
            count += 1 
        else:
            if count > max_count:
                max_count = count
                end_index = i
                count = 0

    return end_index-max_count, end_index-1


def decode_files(model, fname):

    # get names of feature files to decode
    feature_files = get_feature_files(fname)

    # run over all feature files
    left_err = 0
    right_err = 0
    X = []
    Y = []
    for file in feature_files:

        # get feature matrix and the segment size
        fe_matrix = read_features(file) 
        segment_size = fe_matrix.shape[0]

        # get real labels - due to the windows sizes
        real_labels = (LEFT_WINDOW_SIZE, segment_size-RIGHT_WINDOW_SIZE)

        # Predict all the frames at once using the model. 
        # The result is a binary vector indicating whether each time-frame is 
        # predicted as part of the target event or not
        binary_vec = model.predict_sequence(fe_matrix)

        # smooth the binary vector to avoid singular predictions
        smooth_vec = smooth_binary_vector(binary_vec)

        # find indexes of the longest subvector contains only ones
        predicted_labels = find_longest_event(smooth_vec)
        print 'real labels: ', real_labels
        print 'predicted labels: ', predicted_labels

        # store pre-aspiration durations
        X.append(real_labels[1]-real_labels[0])
        Y.append(predicted_labels[1]-predicted_labels[0])

        left_err += np.abs(real_labels[0]-predicted_labels[0])
        right_err += np.abs(real_labels[1]-predicted_labels[1])

    print 'left_err: ',  float(left_err)/len(feature_files)
    print 'right_err: ', float(right_err)/len(feature_files)

    X = np.array(X)
    Y = np.array(Y)

    print "Mean of labeled/predicted preaspiration: %sms, %sms" % (str(np.mean(X)), str(np.mean(Y)))
    print "Standard deviation of labeled/predicted preaspiration: %sms, %sms" % (str(np.std(X)), str(np.std(Y)))
    print "max of labeled/predicted preaspiration: %sms, %sms" % (str(np.max(X)), str(np.max(Y)))
    print "min of labeled/predicted preaspiration: %sms, %sms" % (str(np.min(X)), str(np.min(Y)))


    thresholds = [2, 5, 10, 15, 20, 25, 50]
    print "Percentage of examples with labeled/predicted PA difference of at most:"
    print "------------------------------"
    
    for thresh in thresholds:
        print "%d msec: " % thresh, 100*(len(X[abs(X-Y)<thresh])/float(len(X)))


if __name__ == '__main__':

      # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_names_path", help="A path to a file containing the absolute paths of the feature-files")
    parser.add_argument("params_path", help="A path to a file containing the model parameters (after training)")
    args = parser.parse_args()

    # Construct a model with the pre-trained parameters
    my_model = BiRNNBinaryPredictor(load_from_file=args.params_path)

    # Decode the given files
    decode_files(my_model, args.feature_names_path)