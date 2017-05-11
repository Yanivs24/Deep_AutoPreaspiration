#!/bin/bash

# This file is part of BiRNN_AutoPA - automatic extraction of pre-aspiration 
# from speech segments in audio files.
#
# Copyright (c) 2017 Yaniv Sheena
#
# Usage: 
# decode_after_fe.sh
#
# Description: This script should be called only after successfully performing 
# 			   feature extraction using 'run/extract_features.sh'. This script 
#              decode the testset examples using the extracted features for test.
#			   The model is based on the the tuned parameters (tmp_files/model_params.txt)
#              and hence should be called after training.


# path of the feature file
FEATURE_PATH_TEST=tmp_files/features_test

# path of the model parameters 
MODEL_PARAMS_PATH=tmp_files/model_params.txt

# check if the feature file exists
if [ ! -d $FEATURE_PATH_TEST ]; then
	echo "\"${FEATURE_PATH_TEST}\" - directory not exists"
	exit 1
fi

# check if the parameters file exists
if [ ! -f $MODEL_PARAMS_PATH ]; then
	echo "\"${MODEL_PARAMS_PATH}\" - file not exists - please train a model first"
	exit 2
fi

# decode test examples
python learning/decode_model.py $FEATURE_PATH_TEST $MODEL_PARAMS_PATH

