#!/bin/bash

# This file is part of BiRNN_AutoPA - automatic extraction of pre-aspiration 
# from speech segments in audio files.
#
# Copyright (c) 2017 Yaniv Sheena
#
# Usage: 
# extract_features.sh DATA_PATH NUM_OF_TRAIN_EXAMPLES NUM_OF_TEST_EXAMPLES
#
# Description: 
# Extract features from all the examples in a directory, place the output files
# in the tmp_files/features. Separate the feature files into train and test set 
# according to the given amount. For example, tmp_files/features/feature_names_train.txt 
# will contain the absolute paths of the feature-files of the train set.


# Window sizes (ms)
MIN_WIN_TRAIN=-80
MAX_WIN_TRAIN=80
MIN_WIN_TEST=-50
MAX_WIN_TEST=60

# path for temp config files 
CONFIG_PATH=tmp_files/config
# path for temp feature files 
FEATURE_PATH_TRAIN=tmp_files/features_train
FEATURE_PATH_TEST=tmp_files/features_test

# Input validations 
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo "Usage: $0 DATA_PATH SIZE_OF_TRAIN_SET SIZE_OF_TEST_SET"
    exit 1
fi

examples_path=$1
num_of_samples_train=$2
num_of_samples_test=$3

# Input validation
if [ ! -d $examples_path ]; then
	echo "\"${examples_path}\" - not a directory"
	exit 2
fi

# create directories for the feature-files
rm -rf $FEATURE_PATH_TRAIN
rm -rf $FEATURE_PATH_TEST
mkdir $FEATURE_PATH_TRAIN
mkdir $FEATURE_PATH_TEST
mkdir $FEATURE_PATH_TRAIN/feature_files
mkdir $FEATURE_PATH_TEST/feature_files

# create config directory
rm -rf $CONFIG_PATH
mkdir  $CONFIG_PATH

# Build train&test sets and the corresponding config files using python script
python scripts/build_config_files.py $examples_path $CONFIG_PATH $num_of_samples_train $num_of_samples_test

# call feature extraction with the created config files for train and test sets
python feature_extraction/auto_pa_extract_features.py   \
 --window_min $MIN_WIN_TRAIN 							\
 --window_max $MAX_WIN_TRAIN 							\
 --pa_tier bell 										\
 --pa_mark pre 											\
 $CONFIG_PATH/PreaspirationTrainTgList.txt 				\
 $CONFIG_PATH/PreaspirationTrainWavList.txt 			\
 $FEATURE_PATH_TRAIN/input.txt 							\
 $FEATURE_PATH_TRAIN/feature_names.txt 					\
 $FEATURE_PATH_TRAIN/labels.txt 						\
 $FEATURE_PATH_TRAIN/feature_files

python feature_extraction/auto_pa_extract_features.py   \
--window_min $MIN_WIN_TEST								\
--window_max $MAX_WIN_TEST								\
--pa_tier bell											\
--pa_mark pre 											\
$CONFIG_PATH/PreaspirationTestTgList.txt				\
$CONFIG_PATH/PreaspirationTestWavList.txt				\
$FEATURE_PATH_TEST/input.txt							\
$FEATURE_PATH_TEST/feature_names.txt					\
$FEATURE_PATH_TEST/labels.txt							\
$FEATURE_PATH_TEST/feature_files

