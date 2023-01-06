# This sub-project implements experiments involved in "5.3 Natural Non-IID Conversational Speech Dataset"

## Major Scripts 

`steps_fednp/dataGenSequences*.py`:  data iteraters
`steps_fednp/train*.py`: implementations of the acoustic RNNs and RTN models
`steps_fednp/decode*.sh`:  decoder
`steps_fednp/nnet-forward*.py`:  HMM state posterior probability estimator

## Usage
The running experiment 

Quick intstructions for runing our experiments:

1) Download CHiME5 data

Check the download page of http://spandh.dcs.shef.ac.uk/chime_challenge/CHiME5/download.html

2) Move to Kaldi CHiME5 directory, e.g.,

cd kaldi-trunk/egs/chime5/s5b

3) Copy all files and directories of chime5 into current directory

4) Specify CHiME5 root directory in run.sh e.g.,

chime5_data=<your CHiME5 directory>/CHiME5

5) Execute run_worn.sh

6) Generate Mel-filterbank features

execute steps/make_fbank.sh

7) Generate speaker-level mean and variance normalization

steps/compute_cmvn_stats.sh

8) Split dataset
split_data_fl.sh`: used for creating the federated learning dataset. `./split_data_fl.sh <path-to-fbank> <number-of-clients can be (4, 8 or 16)>`

9) Run the models, you may need to adjust the corresponding parameters in the scripts
SRU FedNP: `./run_sru_fl_fednp.sh`
