#!/usr/bin/python
import numpy as np
import random
import argparse
import sys

FIRST_FEATURE_INDEX = 1
LAST_FEATURE_INDEX  = 9
LEFT_WINDOW_SIZE    = 50
RIGHT_WINDOW_SIZE   = 60


def get_feature_files(feature_names_file):
	with open(feature_names_file) as f:
		file_names = f.readlines()

	return [line.strip() for line in file_names]

def read_features(file_name):
	return np.loadtxt(file_name, skiprows=1)[:, FIRST_FEATURE_INDEX:LAST_FEATURE_INDEX]

def write_examples(data_set, output_path):
	with open(output_path, 'w') as f:
		for x,i,y in data_set:
			f.write("%s,%s,%s\n" % (' '.join(map(str, x)), str(i), str(y)))

def build_dataset(feature_names_path, output_path):
	# get feature files, one file for each example (voice segment)
	print 'Reading feature file-names from %s' % feature_names_path
	feature_files = get_feature_files(feature_names_path)

	print 'Extracting frames from the feature files..'
	data_set = []
	# run over all feature files
	for file in feature_files:

		# get feature matrix and the segment size
		fe_matrix = read_features(file) 
		segment_size = fe_matrix.shape[0]

		full_frame = fe_matrix.flatten()
		# For each time-frame, concatenate 2 vectors of frames to each side. 
		# We represnt each frame with 5 vectors, when the frame's vector is 
		# in the middle (dim is 8*5=40)
		for i in range(segment_size):
			# if the represented frame is within the pre-aspiration range - set label 1
			# and otherwise the label will be 0
			label = 0
			if (i >= LEFT_WINDOW_SIZE) and (i <= (segment_size-RIGHT_WINDOW_SIZE)):
				label = 1

			# append the example to the data set
			data_set.append((full_frame, i, label))


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

	# write data set to file
	print 'Write resulted examples to: "%s"' % output_path
	write_examples(data_set, output_path)


if __name__ == '__main__':

	  # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_names_path", help="A path to a file containing the absolute paths of the feature-files")
    parser.add_argument("output_path", help="The path to the output file")
    args = parser.parse_args()

    build_dataset(args.feature_names_path, args.output_path)

