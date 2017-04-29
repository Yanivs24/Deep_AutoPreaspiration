#!/usr/bin/python
# This file is part of AutoPA - automatic extraction of pre-aspiration 
# from speech segments in audio files.
#
# Copyright (c) 2016 Yaniv Sheena

import os
import sys
import random
from shutil import copyfile, rmtree

def get_examples(dir_path):
    wav_extension = '.wav'
    tg_extension = '.TextGrid'
    wav_files = [n[:-len(wav_extension)] for n in os.listdir(dir_path) if n.endswith(wav_extension)]
    tg_files = [n[:-len(tg_extension)] for n in os.listdir(dir_path) if n.endswith(tg_extension)]

    # intesection between wav files and textgrid (we need both)
    examples = set(wav_files) & set(tg_files)

    return list(examples)


def delete_dirs(dir_paths):
    ''' deletes dir if the dir exists or has premission'''
    for c_dir in dir_paths:
        try:
            rmtree(c_dir)
        except OSError:
            pass


if __name__ == "__main__":

    if len(sys.argv) != 5:
        raise ValueError("expected exactly 4 inputs:\n1)examples dir path\n2)configs dir path \n3)train-set size\n4)test-set size")

    examples_path = sys.argv[1]
    if examples_path.endswith('/'):
        examples_path = examples_path[:-1]

    # change to absolute path
    examples_path = os.path.abspath(examples_path)

        
    config_path = sys.argv[2]
    if not config_path.endswith('/'):
        config_path = config_path + '/'

    train_size = sys.argv[3]
    test_size  = sys.argv[4]

    examples_names = get_examples(examples_path)

    # in order to choose randomly - shuffle list and then take from the beginning
    random.shuffle(examples_names)

    # all feature - prepare just train set or just test set
    if train_size == 'all':
        train_set = examples_names[:]
        test_set  = []
    elif test_size == 'all':
        train_set = []
        test_set  = examples_names[:]
    else:
        train_size  = int(train_size)
        test_size   = int(test_size)
        train_set = examples_names[:train_size]
        test_set  = examples_names[train_size:train_size+test_size]


    wav_train       = []
    wav_train_final = []
    tg_train        = []
    tg_train_final  = []
    wav_test        = []
    wav_test_final  = []
    tg_test         = []
    tg_test_final   = []
    # build train lists
    for ex_name in train_set:
        wav_train.append('{0}/{1}.wav'.format(examples_path, ex_name))
        tg_train.append('{0}/{1}.TextGrid'.format(examples_path, ex_name))
        wav_train_final.append('{0}/Train/{1}.wav'.format(examples_path, ex_name))
        tg_train_final.append('{0}/Train/{1}.TextGrid'.format(examples_path, ex_name))

    # build test lists
    for ex_name in test_set:
        wav_test.append('{0}/{1}.wav'.format(examples_path, ex_name))
        tg_test.append('{0}/{1}.TextGrid'.format(examples_path, ex_name))
        wav_test_final.append('{0}/Test/{1}.wav'.format(examples_path, ex_name))
        tg_test_final.append('{0}/Test/{1}.TextGrid'.format(examples_path, ex_name))

    # delete train and test dirs (if existed) and create new ones
    train_dst_dir = '%s/Train' % examples_path
    test_dst_dir = '%s/Test'   % examples_path
    delete_dirs([train_dst_dir, test_dst_dir])
    if train_set:
        os.mkdir(train_dst_dir)
    if test_set:
        os.mkdir(test_dst_dir)

    # copy files to Train and Test dirs
    for file_path in wav_train+tg_train:
        print 'copy %s to the train directory' % file_path
        copyfile(file_path, os.path.join(train_dst_dir, os.path.split(file_path)[1]))

    for file_path in wav_test+tg_test:
        print 'copy %s to the test directory' % file_path
        copyfile(file_path, os.path.join(test_dst_dir, os.path.split(file_path)[1]))

    # Build the files that contain paths lists as needed and place them in 'config' dir 
    if train_set:
        with open(config_path+'PreaspirationTrainWavList.txt', 'w') as f:
            f.write('\n'.join(wav_train_final))

        with open(config_path+'PreaspirationTrainTgList.txt', 'w') as f:
            f.write('\n'.join(tg_train_final))

    if test_set:
        with open(config_path+'PreaspirationTestWavList.txt', 'w') as f:
            f.write('\n'.join(wav_test_final))

        with open(config_path+'PreaspirationTestTgList.txt', 'w') as f:
            f.write('\n'.join(tg_test_final))




