#!/usr/bin/python

# This file is part of BiRNN_AutoPA - automatic extraction of pre-aspiration 
# from speech segments in audio files.
#
# Copyright (c) 2017 Yaniv Sheena
#
# Description: 
# format all wav files in given dir to 16kHz mono and place them in a new sub-dir

import os
import sys
from shutil import copyfile, rmtree

def get_files_with_extension(dir_path, extention):
    return [os.path.join(dir_path, f_name) for f_name in os.listdir(dir_path) if f_name.endswith(extention)]


def format_wav_files(wav_list, dst_dir):
    ''' format wav files listed in 'wav_list' to the needed format for training and 
        place the new wav-files in 'dst_dir' without spaces in their names'''

    # Delete old dst_dir
    try:
        rmtree(dst_dir)
    except OSError:
        pass

    # now we can create the new dir
    os.mkdir(dst_dir)

    # Format files using 'sox'
    for file in wav_list:
        file_name_no_spaces = (os.path.split(file)[1]).replace(' ', '_')
        print 'formating %s and saving new file as:\nformated/%s' % (file, file_name_no_spaces)
        os.system("sox '%s' -c 1 -r 16000 '%s/%s'" % (file, dst_dir, file_name_no_spaces))

if __name__ == "__main__":
    # Assuming the first cmd-line argument is the path of
    # the directory with the original wav files.

    if len(sys.argv) != 2:
        raise ValueError("expected exactly one argument - examples dir's path")

    src_dir_path = sys.argv[1]

    # get a list of wav files paths
    wav_files_list = get_files_with_extension(src_dir_path, '.wav')
    dst_dir_path = os.path.join(src_dir_path, 'formated')

    # format all files using sox and place them in 'dst_dir' (with the same names)
    format_wav_files(wav_files_list, dst_dir_path)

    # now copy also "TextGrid" files for simplicity
    textgrid_files = get_files_with_extension(src_dir_path, '.TextGrid')
    print 'Now copying TextGrid files...'
    for tg_file in textgrid_files:
        file_name_no_spaces = (os.path.split(tg_file)[1]).replace(' ', '_')
        copyfile(tg_file, dst_dir_path+'/'+ file_name_no_spaces)
    print 'Completed!'
