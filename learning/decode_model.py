import numpy as np
import dynet as dy
import argparse
import os

from model.model import BiRNNBinaryPredictor

FILE_WITH_FEATURE_FILE_LIST = 'feature_names.txt'
FILE_WITH_LABELS            = 'labels.txt'
FIRST_FEATURE_INDEX = 1
LAST_FEATURE_INDEX  = 9
NUM_OF_FEATURES_PER_FRAME = LAST_FEATURE_INDEX-FIRST_FEATURE_INDEX


def get_feature_files(feature_path):
    full_path = os.path.join(feature_path, FILE_WITH_FEATURE_FILE_LIST)
    with open(full_path) as f:
        file_names = f.readlines()

    return [line.strip() for line in file_names]

def get_labels(feature_path):
    full_path = os.path.join(feature_path, FILE_WITH_LABELS)
    with open(full_path) as f:
        file_labels = f.readlines()

    return [map(int, line.strip().split()) for line in file_labels[1:]]

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


def decode_files(model, feature_path):

    # get names of feature files to decode
    feature_files_list = get_feature_files(feature_path)

    # get their corresponding labels
    labels_list = get_labels(feature_path)

    # run over all feature files
    left_err = 0
    right_err = 0
    X = []
    Y = []
    for file, labels in zip(feature_files_list, labels_list):

        # get feature matrix and the segment size
        fe_matrix = read_features(file) 
        segment_size = fe_matrix.shape[0]

        # get real labels - due to the windows sizes
        #labels = (LEFT_WINDOW_SIZE, segment_size-RIGHT_WINDOW_SIZE)

        # Predict all the frames at once using the model. 
        # The result is a binary vector indicating whether each time-frame is 
        # predicted as part of the target event or not
        binary_vec = model.predict_sequence(fe_matrix)

        # smooth the binary vector to avoid singular predictions
        smooth_vec = smooth_binary_vector(binary_vec)

        # find indexes of the longest subvector contains only ones
        predicted_labels = find_longest_event(smooth_vec)
        print 'real labels: ', labels
        print 'predicted labels: ', predicted_labels

        # store pre-aspiration durations
        X.append(labels[1]-labels[0])
        Y.append(predicted_labels[1]-predicted_labels[0])

        # not found - zeros vector
        if predicted_labels[1] <= predicted_labels[0]:
            print 'Warning - event has not found in: %s' % file
            

        left_err += np.abs(labels[0]-predicted_labels[0])
        right_err += np.abs(labels[1]-predicted_labels[1])

    print 'left_err: ',  float(left_err)/len(feature_files_list)
    print 'right_err: ', float(right_err)/len(feature_files_list)

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
    parser.add_argument("feature_path", help="A path to a directory containing the extracted feature-files and the labels")
    parser.add_argument("params_path", help="A path to a file containing the model parameters (after training)")
    args = parser.parse_args()

    # Construct a model with the pre-trained parameters
    my_model = BiRNNBinaryPredictor(load_from_file=args.params_path)

    # Decode the given files
    decode_files(my_model, args.feature_path)