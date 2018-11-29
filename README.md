# Deep AutoPreaspiration
Automatic extraction of pre-aspiration from voice segments using bi-directional RNN based method


It works as follows:
* The user provides wav files containing one (or more) obstruents, and corresponding Praat TextGrids containing some information about roughly where the pre-aspiration should be (e.g. time index within the preceding phoneme and time index within the obstruent).
* A classifier is used to find the pre-aspiration, for each coded obstruent, and output the boundaries of the pre-aspiration.
* The user can either use a pre-existing classifier, or (recommended) train a new one using a manually-labeled pre-aspirations from their own data.

## Installation

The code to clone AutoPreaspiration is: 

	$ git clone https://github.com/Yanivs24/BiRNN_AutoPreaspiration.git
	

Alternatively, you can download the current version of AutoPreaspiration as a zip file, just press "Clone or download" -> "Download ZIP"

## Setup
First, navigate to the project root directory.

In order to work with a new data, the wav files should be converted to 16kHz mono.
This can be done by typing (***A prerequisite for this script is installing SoX utility***):

	$ python scripts/format_wav_files.py DIRECTORY_PATH

When 'DIRECTORY_PATH' is the path of the directory containing the data (wav&TextGrids)
This will place all the formatted wav files and the corresponding TextGrids in: DIRECTORY_PATH/formated

Now, the examples should be filtered using:

	$ python scripts/filter_examples.py DIRECTORY_PATH/formated/
  
This script is responsible for two things:
1) Deletes all the examples without a 'pre' mark in their last tier in the TextGrid file, or with
   'pre' mark but without windows big enough for the algorithm.
2) If the last tier containing 'pre', it changes its name to 'bell' for uniformity.


## Usage
The bash scripits in the directory 'run/' wraps the logic of the project with a vey simple user interface.

### Training 
To train a model using all the files (wav and corresponding TextGrid) in a directory type:

	$ run/train_dir.sh DIRECTORY_PATH
  
For example:	

	$ run/train_dir.sh data/abe24_abe18_plosives/formated/
  
### Decoding
To decode all the examples in a directory using a pre-trained model type:

	$ run/decode_dir.sh DIRECTORY_PATH
  
Note: This can be used only after training a model

### Feature extraction and test mode
To extract features from files in a directory, randomly splitting them into
training set and test set use:

	$ run/extract_features.sh DATA_PATH SIZE_OF_TRAIN_SET SIZE_OF_TEST_SET

##### SIZE_OF_TRAIN_SET is the number of examples to be used as a training set (type 'all' for all the examples in the directory)
##### SIZE_OF_TEST_SET is the number of examples to be used as a test set (type 'all' for all the examples in the directory)

Now, after extracting features, we can simply train a new model using the extracted training set:

	$ run/train_after_fe.sh
  
And we can test the performance of the new model on the extracted test set using:

	$ run/decode_after_fe.sh


  

