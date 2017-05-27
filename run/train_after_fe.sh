#!/bin/bash

# This file is part of BiRNN_AutoPA - automatic extraction of pre-aspiration 
# from speech segments in audio files.
#
# Copyright (c) 2017 Yaniv Sheena
#
# Usage: 
# train_after_fe.sh
#
# Description: 
# This script should be called only after successfully performing 
# feature extraction using 'run/extract_features.sh'. This script trains
# a model using the extracted features for the training set, and store the
# tuned parameters in a file that can be used by the decoder (tmp_files/model_params.txt).


# Path of the feature file
FEATURE_PATH_TRAIN=tmp_files/features_train

# Path of the dataset that will be built for the trainer
OUTPUT_DATASET_PATH=tmp_files/dataset

# Path of the model parameters - after the training is finished
MODEL_PARAMS_PATH=tmp_files/model_params.txt

# Check if the feature directory exists
if [ ! -d $FEATURE_PATH_TRAIN ]; then
    echo "\"${FEATURE_PATH_TRAIN}\" - directory not exists"
    exit 1
fi

# Build dataset from features
python scripts/create_dataset.py $FEATURE_PATH_TRAIN $OUTPUT_DATASET_PATH

# Check exit code
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "create_dataset.py failed with exit code $exit_code"
    exit $exit_code
fi

# Train the model using the built dataset - the parameters will be stored in
# 'MODEL_PARAMS_PATH'
python learning/train_model.py $OUTPUT_DATASET_PATH $MODEL_PARAMS_PATH
