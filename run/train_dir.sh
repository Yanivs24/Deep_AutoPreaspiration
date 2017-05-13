#!/bin/bash

# This file is part of BiRNN_AutoPA - automatic extraction of pre-aspiration 
# from speech segments in audio files.
#
# Copyright (c) 2017 Yaniv Sheena
#
# Usage: 
# train_dir.sh DIRECTORY_PATH
#
# Description: 
# Train a model using all the examples within the given path. The tuned parameters
# are stored a file that can be used by the decoder (tmp_files/model_params.txt).
# Note: The examples should be formated using "python_scripts/format_wav_files.py" 
#       and filtered using "python_scripts/filter_examples.py".


# Window sizes (ms)
MIN_WIN_TRAIN=-80
MAX_WIN_TRAIN=80
# Path for temp config files for feature extraction
CONFIG_PATH=tmp_files/config
# Path for temp feature files 
FEATURE_PATH=tmp_files/features_train
# Path of the dataset that will be built for the trainer
OUTPUT_DATASET_PATH=tmp_files/dataset
# Path of the model parameters 
MODEL_PARAMS_PATH=tmp_files/model_params.txt

# Validate arguments amount
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: $0 DIRECTORY_PATH"
    exit 1
fi

examples_path=$1

# Input validation 
if [ ! -d $examples_path ]; then
    echo "\"${examples_path}\" - not a directory"
    exit 1
fi

# Empty config directory
rm -rf $CONFIG_PATH
mkdir  $CONFIG_PATH

# Empty features directory
rm -rf $FEATURE_PATH
mkdir  $FEATURE_PATH
mkdir  $FEATURE_PATH/feature_files

# Build training set and the corresponding config files using python script
# the training set is built from all the examples in the given directory (examples_path)
python scripts/build_config_files.py $examples_path $CONFIG_PATH all 0

# Check exit code
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "build_config_files.py failed with exit code $exit_code"
    exit $exit_code
fi

# Call feature extraction with the created config file - this should
# extract the features from the files and placed them in 'FEATURE_PATH'
python feature_extraction/auto_pa_extract_features.py       \
 --window_min $MIN_WIN_TRAIN                                \
 --window_max $MAX_WIN_TRAIN                                \
 --pa_tier bell                                             \
 --pa_mark pre                                              \
 $CONFIG_PATH/PreaspirationTrainTgList.txt                  \
 $CONFIG_PATH/PreaspirationTrainWavList.txt                 \
 $FEATURE_PATH/input.txt                                    \
 $FEATURE_PATH/feature_names.txt                            \
 $FEATURE_PATH/labels.txt                                   \
 $FEATURE_PATH/feature_files

# Check exit code
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "auto_pa_extract_features.py failed with exit code $exit_code"
    exit $exit_code
fi

# Build dataset from features
python scripts/create_dataset.py $FEATURE_PATH $OUTPUT_DATASET_PATH

# Check exit code
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "create_dataset.py failed with exit code $exit_code"
    exit $exit_code
fi

# Train the model using the built dataset - the parameters will be stored in
# 'MODEL_PARAMS_PATH'
python learning/train_model.py $OUTPUT_DATASET_PATH $MODEL_PARAMS_PATH






