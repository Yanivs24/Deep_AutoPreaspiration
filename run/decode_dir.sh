#!/bin/bash

# This file is part of BiRNN_AutoPA - automatic extraction of pre-aspiration 
# from speech segments in audio files.
#
# Copyright (c) 2017 Yaniv Sheena
#
# Usage: 
# decode_dir.sh DIRECTORY_PATH
#
# Description: 
# Decode the examples in the given directory (DIRECTORY_PATH) and test the performance by
# comparing the predictions to the manually coded pre-aspiration (within the textgrids).
# Note: The examples should be formated using "scripts/format_wav_files.py" 
# and filtered using "scripts/filter_examples.py"


# Window sizes (ms)
MIN_WIN=-50
MAX_WIN=60
# Path for temp config files for feature extraction
CONFIG_PATH=tmp_files/config
# Path for temp feature files 
FEATURE_PATH=tmp_files/features_infer
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

# Params file should be exists
if [ ! -f $MODEL_PARAMS_PATH ]; then
    echo "\"${MODEL_PARAMS_PATH}\" - file not exists - please train a model first"
    exit 2
fi

# Empty config directory
rm -rf $CONFIG_PATH
mkdir  $CONFIG_PATH

# Empty features directory
rm -rf $FEATURE_PATH
mkdir  $FEATURE_PATH
mkdir  $FEATURE_PATH/feature_files

# Build test set and the corresponding config files using python script
# using the examples in the given directory
python scripts/build_config_files.py $examples_path $CONFIG_PATH 0 all

# Check exit code
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "build_config_files.py failed with exit code $exit_code"
    exit $exit_code
fi

# Call feature extraction with the created config file - this should
# extract the features from the files and placed them in FEATURE_PATH
python feature_extraction/auto_pa_extract_features.py       \
 --window_min $MIN_WIN                                      \
 --window_max $MAX_WIN                                      \
 --pa_tier bell                                             \
 --pa_mark pre                                              \
 $CONFIG_PATH/PreaspirationTestTgList.txt                   \
 $CONFIG_PATH/PreaspirationTestWavList.txt                  \
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

# Decode files
python learning/decode_model.py $FEATURE_PATH $MODEL_PARAMS_PATH






