#!/usr/bin/python
import numpy as np
import random
import argparse
import sys
import os

FILE_WITH_FEATURE_FILE_LIST = 'feature_names.txt'
FILE_WITH_LABELS            = 'labels.txt'
FIRST_FEATURE_INDEX = 1
LAST_FEATURE_INDEX  = 9
#LEFT_WINDOW_SIZE    = 80
#RIGHT_WINDOW_SIZE   = 80
WINDOW_STD          = 20


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

def write_examples(data_set, output_path):
    with open(output_path, 'w') as f:
        for x,i,y in data_set:
            f.write("%s,%s,%s\n" % (' '.join(map(str, x)), str(i), str(y)))

def build_dataset(feature_path, output_path):
    # get feature files, one file for each example (voice segment)
    print 'Reading features and labels from %s' % feature_path

    # get names of feature files 
    feature_files_list = get_feature_files(feature_path)

    # get their corresponding labels
    labels_list = get_labels(feature_path)

    print 'Extracting frames from the feature files..'
    data_set = []
    # run over all feature files
    for file, labels in zip(feature_files_list, labels_list):

        # get feature matrix and the segment size
        fe_matrix = read_features(file) 
        full_segment_size = fe_matrix.shape[0]

        # Choose the size of the windows surrounding the event randomly - this is done in order to 
        # avoid a situation in which the constant length of the windows affects the BiRNN (the RNN can learn
        # things from the length of the windows)
        left_index  = max(0, int(np.random.normal(0.5*labels[0], WINDOW_STD)))
        right_index = min(full_segment_size, int(np.random.normal(0.5*(full_segment_size+labels[1]), WINDOW_STD)))

        if left_index >= right_index:
            print 'skipping example - too short'
            continue

        # debug
        print left_index, right_index

        # Crop the segment using the random window-sizes
        # print 'Croping segment to %s:%s' % (left_index, right_index)
        cur_frame = fe_matrix[left_index:right_index, :].flatten()

        # For each time-frame in the cropped segment, set it binary label and add the point (frame,index,lael)
        # to the dataset
        for i in range(right_index-left_index):
            abs_index = left_index + i
            # if the represented frame is within the pre-aspiration range - set label 1
            # and otherwise the label will be 0
            label = 0
            if (abs_index >= labels[0]) and (abs_index <= labels[1]):
                label = 1

            # append the example to the data set
            data_set.append((cur_frame, i, label))


    # since there are much more frames that are not part of pre-aspiration event (negative examples),
    # we balance the data set by randomly dropping such negative examples.
    positive_size = sum(example[2]==1 for example in data_set)
    negative_drop_amount = len(data_set) - 2*positive_size

    # shuffle data 
    random.shuffle(data_set)

    ind = 0
    dropped_amount = 0
    # Remove first #negative_drop_amount negative examples
    while dropped_amount < negative_drop_amount:
        if data_set[ind][2] == 0:
            data_set.pop(ind)
            dropped_amount += 1
        else:
            ind += 1

    # shuffle data again
    random.shuffle(data_set)

    print '%s examples were extracted from the files, %s%% are positive' % (len(data_set), 100.0*positive_size/len(data_set))

    # write data set to file
    print 'Write examples to: "%s"' % output_path
    write_examples(data_set, output_path)


if __name__ == '__main__':

      # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_path", help="A path to a directory containing the extracted feature-files and the labels")
    parser.add_argument("output_path", help="The path to the output file")
    args = parser.parse_args()

    build_dataset(args.feature_path, args.output_path)

