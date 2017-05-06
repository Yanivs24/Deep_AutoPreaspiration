#!/bin/bash

# This file is part of BiRNN_AutoPA - automatic extraction of pre-aspiration 
# from speech segments in audio files.
#
# Copyright (c) 2016 Yaniv Sheena
#
# Usage: 
# train_after_fe.sh
#
# Description: This script should be called only after successfully performing 
# 			   feature extraction using 'run/extract_features.sh'. This script trains
#			   a model using the extracted features, and store the tuned parameters
#			   in a file that can be used by the decoder.


# path of the feature file
TRAIN_FEATURES_PATH=tmp_files/features/feature_names_train.txt

# path of the dataset that will be built for the trainer
OUTPUT_DATASET_PATH=tmp_files/dataset

# path of the model parameters - after the training is finished
MODEL_PARAMS_PATH=tmp_files/model_params.txt

# check if feature file exists
if [ ! -f $TRAIN_FEATURES_PATH ]; then
	echo "\"${TRAIN_FEATURES_PATH}\" - file not exists"
	exit 1
fi

# build dataset from features
python scripts/create_dataset.py $TRAIN_FEATURES_PATH $OUTPUT_DATASET_PATH

# train the model
python learning/train_model.py $OUTPUT_DATASET_PATH $MODEL_PARAMS_PATH
