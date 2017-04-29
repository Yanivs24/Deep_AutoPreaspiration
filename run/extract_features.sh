#!/bin/bash

# This file is part of BiRNN_AutoPA - automatic extraction of pre-aspiration 
# from speech segments in audio files.
#
# Copyright (c) 2016 Yaniv Sheena
#
# Usage: 
# extract_features.sh DATA_PATH NUM_OF_TRAIN_EXAMPLES NUM_OF_TEST_EXAMPLES
#
# Description: extract features from all the examples in a directory, place the output files
#			   in the tmp_files/features. Separate the feature files into train and test set 
#  			   according to the given amount. For example, tmp_files/features/feature_names_train.txt 
# 			   will contain the absolute paths of the feature-files of the train set.


min_val=-50
max_val=60
examples_path=$1
num_of_samples_train=$2
num_of_samples_test=$3

# path for temp config files 
config_path=tmp_files/config
# path for temp feature files 
feature_path=tmp_files/features

# create directories for the feature-files
rm -rf $feature_path
mkdir $feature_path
mkdir $feature_path/feature_files_train
mkdir $feature_path/feature_files_test

# create config directory
rm -rf $config_path
mkdir  $config_path

# Build train-set and the corresponding config files using python script
python scripts/build_config_files.py $examples_path $config_path $num_of_samples_train $num_of_samples_test

# call feature extraction with the created config files for train and test sets
feature_extraction/auto_pa_extract_features.py --window_min $min_val --window_max $max_val --pa_tier bell --pa_mark pre $config_path/PreaspirationTrainTgList.txt $config_path/PreaspirationTrainWavList.txt $feature_path/input_train.txt $feature_path/feature_names_train.txt $feature_path/labels_train.txt $feature_path/feature_files_train
feature_extraction/auto_pa_extract_features.py --window_min $min_val --window_max $max_val --pa_tier bell --pa_mark pre $config_path/PreaspirationTestTgList.txt $config_path/PreaspirationTestWavList.txt $feature_path/input_test.txt $feature_path/feature_names_test.txt $feature_path/labels_test.txt $feature_path/feature_files_test
