#!/bin/bash
#
# Based mostly on the TED-LIUM and Switchboard recipe


# Begin configuration section.
nj=12
decode_nj=12
stage=21
                       # change this variable and stage 4
# End configuration section
. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh


set -e # exit on error

# chime5 main directory path
# please change the path accordingly
chime5_corpus=CHiME5
json_dir=${chime5_corpus}/transcriptions
audio_dir=${chime5_corpus}/audio

# training and test data
train_set=train_worn
test_sets="dev_worn eval_worn" #"dev_worn dev_addition_dereverb_ref"

# This script also needs the phonetisaurus g2p, srilm, beamformit
./local/check_tools.sh || exit 1


if [ $stage -le 1 ]; then
  # skip u03 as they are missing
  for mictype in worn ; do
    local/prepare_data.sh --mictype ${mictype} \
			  ${audio_dir}/train ${json_dir}/train data/train_${mictype}
  done
  for dataset in dev eval ; do
    for mictype in worn; do
      local/prepare_data.sh --mictype ${mictype} \
			    ${audio_dir}/${dataset} ${json_dir}/${dataset} \
			    data/${dataset}_${mictype}
    done
  done
fi

if [ $stage -le 2 ]; then
  local/prepare_dict.sh

  utils/prepare_lang.sh \
    data/local/dict "<unk>" data/local/lang data/lang

  local/train_lms_srilm.sh \
    --train-text data/train_worn/text --dev-text data/dev_worn/text \
    --oov-symbol "<unk>" --words-file data/lang/words.txt \
    data/ data/srilm
fi

LM=data/srilm/best_3gram.gz
if [ $stage -le 3 ]; then
  # Compiles G for chime5 trigram LM
  utils/format_lm.sh \
		data/lang $LM data/local/dict/lexicon.txt data/lang

fi


if [ $stage -le 5 ]; then
  # remove possibly bad sessions (P11_S03, P52_S19, P53_S24, P54_S24)
  # see http://spandh.dcs.shef.ac.uk/chime_challenge/data.html for more details
  utils/copy_data_dir.sh data/train_worn data/train_worn_org # back up
  grep -v -e "^P11_S03" -e "^P52_S19" -e "^P53_S24" -e "^P54_S24" data/train_worn_org/text > data/train_worn/text
  utils/fix_data_dir.sh data/train_worn
fi


if [ $stage -le 7 ]; then

  # only use left channel for worn mic recognition
  # you can use both left and right channels for training
  for dset in train dev eval; do
    utils/copy_data_dir.sh data/${dset}_worn data/${dset}_worn_stereo
    grep "\.L-" data/${dset}_worn_stereo/text > data/${dset}_worn/text
    utils/fix_data_dir.sh data/${dset}_worn
  done
fi

if [ $stage -le 8 ]; then
  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  for dset in train_worn  dev_worn eval_worn ; do
    utils/copy_data_dir.sh data/${dset} data/${dset}_nosplit
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}_nosplit data/${dset}
  done
fi

if [ $stage -le 8 ]; then
  # Now make MFCC features.
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  mfccdir=mfcc
  for x in train_worn dev_worn eval_worn ; do
    steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" \
		       data/$x exp/make_mfcc/$x $mfccdir
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
    utils/fix_data_dir.sh data/$x
  done
fi


if [ $stage -le 9 ]; then
  mkdir -p data-fbank
  cd exp
  mkdir -p make_fbank
  cd ..
  fbankdir=fbank
  for x in train_worn dev_worn eval_worn; do
  cp -r data/$x data-fbank/$x
    steps/make_fbank.sh --nj 20 --cmd "$train_cmd" \
      data-fbank/$x exp/make_fbank/$x $fbankdir
    steps/compute_cmvn_stats.sh data-fbank/$x exp/make_fbank/$x $fbankdir
    utils/fix_data_dir.sh data/$x
  done
fi











if [ $stage -le 10 ]; then
  # make a subset for monophone training
  utils/subset_data_dir.sh --shortest data/${train_set} 30000 data/${train_set}_30kshort
fi

if [ $stage -le 10 ]; then
  # Starting basic training on MFCC features
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
		      data/${train_set}_30kshort data/lang exp/mono
fi

if [ $stage -le 11 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
		    data/${train_set} data/lang exp/mono exp/mono_ali

  steps/train_deltas.sh --cmd "$train_cmd" \
			2500 30000 data/${train_set} data/lang exp/mono_ali exp/tri1
fi

if [ $stage -le 12 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
		    data/${train_set} data/lang exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
			  4000 50000 data/${train_set} data/lang exp/tri1_ali exp/tri2
fi

if [ $stage -le 13 ]; then
  utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph
  for dset in ${test_sets}; do
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
		    exp/tri2/graph data/${dset} exp/tri2/decode_${dset} &
  done
  wait
fi

if [ $stage -le 14 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
		    data/${train_set} data/lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd "$train_cmd" \
		     5000 100000 data/${train_set} data/lang exp/tri2_ali exp/tri3
fi

if [ $stage -le 15 ]; then
  utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph
  for dset in ${test_sets}; do
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
			  exp/tri3/graph data/${dset} exp/tri3/decode_${dset} &
  done
  wait
fi

if [ $stage -le 16 ]; then
  # The following script cleans the data and produces cleaned data
  steps/cleanup/clean_and_segment_data.sh --nj ${nj} --cmd "$train_cmd" \
    --segmentation-opts "--min-segment-length 0.3 --min-new-segment-length 0.6" \
    data/${train_set} data/lang exp/tri3 exp/tri3_cleaned data/${train_set}_cleaned
fi







# get alignments and prepare fbank for train_worn_cleaned
if [ $stage -le 17 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train_worn_cleaned data/lang exp/tri3_cleaned exp/tri3_cleaned_ali
  steps/align_fmllr.sh --nj 4 --cmd "$train_cmd" \
    data/dev_worn data/lang exp/tri3_cleaned exp/tri3_cleaned_ali_dev







  fbankdir=fbank
  for x in train_worn_cleaned; do
  cp -r data/$x data-fbank/$x
    steps/make_fbank.sh --nj 20 --cmd "$train_cmd" \
      data-fbank/$x exp/make_fbank/$x $fbankdir
    steps/compute_cmvn_stats.sh data-fbank/$x exp/make_fbank/$x $fbankdir
    utils/fix_data_dir.sh data/$x
  done

fi






# use data_fmllr variable for convinience
gmmdir=exp/tri3_cleaned_ali
data_fmllr=data-fbank


# pre-train dnn
dir=exp/tri4a_dnn_pretrain
mkdir -p $dir
if [ $stage -le 18 ]; then
  echo "[INFO] starting from stage 18"
  $cuda_cmd $dir/_pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --nn-depth 7 --rbm-iter 3 $data_fmllr/train_worn_cleaned $dir
fi

# train dnn
dir=exp/tri4a_dnn
ali=exp/tri3_cleaned_ali
ali_dev=exp/tri3_cleaned_ali_dev
feature_transform=exp/tri4a_dnn_pretrain/final.feature_transform
dbn=exp/tri4a_dnn_pretrain/7.dbn
if [ $stage -le 19 ]; then
  $cuda_cmd $dir/_train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/train_worn_cleaned $data_fmllr/dev_worn data/lang $ali $ali_dev $dir
fi

if [ $stage -le 20 ]; then
  utils/mkgraph.sh data/lang $dir $dir/graph
  steps/nnet/decode.sh --nj 1  --cmd "$decode_cmd" --acwt 0.0833 --config conf/decode_dnn.config \
    $dir/graph $data_fmllr/eval_worn $dir/decode_eval &
  steps/nnet/decode.sh --nj 1  --cmd "$decode_cmd" --acwt 0.0833 --config conf/decode_dnn.config \
    $dir/graph $data_fmllr/dev_worn $dir/decode_dev &


fi
gmm=exp/tri4a_dnn
dir=exp/tri5a_dnn
ali=exp/tri4a_dnn_ali
ali_dev=exp/tri4a_dnn_ali_dev
feature_transform=exp/tri4a_dnn_pretrain/final.feature_transform
dbn=exp/tri4a_dnn_pretrain/7.dbn
if [ $stage -le 21 ]; then
  steps/nnet/align.sh --nj 12 --cmd "$train_cmd" \
    $data_fmllr/train_worn_cleaned data/lang $gmm ${gmm}_ali
  steps/nnet/align.sh --nj 12 --cmd "$train_cmd" \
    $data_fmllr/dev_worn data/lang $gmm ${gmm}_ali_dev
  # steps/nnet/make_denlats.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.0833
fi



if [ $stage -le 22 ]; then
  $cuda_cmd $dir/_train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/train_worn_cleaned $data_fmllr/dev_worn data/lang $ali $ali_dev $dir
fi

if [ $stage -le 23 ]; then
  utils/mkgraph.sh data/lang $dir $dir/graph
  steps/nnet/decode.sh --nj 6  --cmd "$decode_cmd" --acwt 0.0833 --config conf/decode_dnn.config \
    $dir/graph $data_fmllr/eval_worn $dir/decode_eval &
  steps/nnet/decode.sh --nj 6  --cmd "$decode_cmd" --acwt 0.0833 --config conf/decode_dnn.config \
    $dir/graph $data_fmllr/dev_worn $dir/decode_dev &
fi



gmm=exp/tri5a_dnn
dir=exp/tri6a_dnn
ali=exp/tri5a_dnn_ali
ali_dev=exp/tri5a_dnn_ali_dev
feature_transform=exp/tri4a_dnn_pretrain/final.feature_transform
dbn=exp/tri4a_dnn_pretrain/7.dbn
if [ $stage -le 24 ]; then
  steps/nnet/align.sh --nj 12 --cmd "$train_cmd" \
    $data_fmllr/train_worn_cleaned data/lang $gmm ${gmm}_ali
  steps/nnet/align.sh --nj 12 --cmd "$train_cmd" \
    $data_fmllr/dev_worn data/lang $gmm ${gmm}_ali_dev

  #steps/nnet/make_denlats.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.0833
fi



if [ $stage -le 25 ]; then
  $cuda_cmd $dir/_train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/train_worn_cleaned $data_fmllr/dev_worn data/lang $ali $ali_dev $dir
fi

if [ $stage -le 26 ]; then
  utils/mkgraph.sh data/lang $dir $dir/graph
  steps/nnet/decode.sh --nj 6  --cmd "$decode_cmd" --acwt 0.0833 --config conf/decode_dnn.config \
    $dir/graph $data_fmllr/eval_worn $dir/decode_eval &
  steps/nnet/decode.sh --nj 6  --cmd "$decode_cmd" --acwt 0.0833 --config conf/decode_dnn.config \
    $dir/graph $data_fmllr/dev_worn $dir/decode_dev &
fi




gmm=exp/tri6a_dnn
dir=exp/tri7a_dnn
ali=exp/tri6a_dnn_ali
ali_dev=exp/tri6a_dnn_ali_dev
feature_transform=exp/tri4a_dnn_pretrain/final.feature_transform
dbn=exp/tri4a_dnn_pretrain/7.dbn
if [ $stage -le 27 ]; then
  steps/nnet/align.sh --nj 12 --cmd "$train_cmd" \
    $data_fmllr/train_worn_cleaned data/lang $gmm ${gmm}_ali
  steps/nnet/align.sh --nj 12 --cmd "$train_cmd" \
    $data_fmllr/dev_worn data/lang $gmm ${gmm}_ali_dev
  #steps/nnet/make_denlats.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.0833
fi



if [ $stage -le 28 ]; then
  $cuda_cmd $dir/_train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/train_worn_cleaned $data_fmllr/dev_worn data/lang $ali $ali_dev $dir
fi

if [ $stage -le 29 ]; then
  utils/mkgraph.sh data/lang $dir $dir/graph
  steps/nnet/decode.sh --nj 6  --cmd "$decode_cmd" --acwt 0.0833 --config conf/decode_dnn.config \
    $dir/graph $data_fmllr/eval_worn $dir/decode_eval &
  steps/nnet/decode.sh --nj 6  --cmd "$decode_cmd" --acwt 0.0833 --config conf/decode_dnn.config \
    $dir/graph $data_fmllr/dev_worn $dir/decode_dev &
fi



gmm=exp/tri7a_dnn
dir=exp/tri8a_dnn
ali=exp/tri7a_dnn_ali
ali_dev=exp/tri7a_dnn_ali_dev
feature_transform=exp/tri4a_dnn_pretrain/final.feature_transform
dbn=exp/tri4a_dnn_pretrain/7.dbn
if [ $stage -le 30 ]; then
  steps/nnet/align.sh --nj 12 --cmd "$train_cmd" \
    $data_fmllr/train_worn_cleaned data/lang $gmm ${gmm}_ali
  steps/nnet/align.sh --nj 12 --cmd "$train_cmd" \
    $data_fmllr/dev_worn data/lang $gmm ${gmm}_ali_dev
  #steps/nnet/make_denlats.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.0833
fi



if [ $stage -le 31 ]; then
  $cuda_cmd $dir/_train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/train_worn_cleaned $data_fmllr/dev_worn data/lang $ali $ali_dev $dir
fi

if [ $stage -le 32 ]; then
  utils/mkgraph.sh data/lang $dir $dir/graph
  steps/nnet/decode.sh --nj 6  --cmd "$decode_cmd" --acwt 0.0833 --config conf/decode_dnn.config \
    $dir/graph $data_fmllr/eval_worn $dir/decode_eval &
  steps/nnet/decode.sh --nj 6  --cmd "$decode_cmd" --acwt 0.0833 --config conf/decode_dnn.config \
    $dir/graph $data_fmllr/dev_worn $dir/decode_dev &
fi

gmm=exp/tri8a_dnn
dir=exp/tri9a_dnn
ali=exp/tri8a_dnn_ali
ali_dev=exp/tri8a_dnn_ali_dev
feature_transform=exp/tri4a_dnn_pretrain/final.feature_transform
dbn=exp/tri4a_dnn_pretrain/7.dbn
if [ $stage -le 33 ]; then
  steps/nnet/align.sh --nj 12 --cmd "$train_cmd" \
    $data_fmllr/train_worn_cleaned data/lang $gmm ${gmm}_ali
  steps/nnet/align.sh --nj 12 --cmd "$train_cmd" \
    $data_fmllr/dev_worn data/lang $gmm ${gmm}_ali_dev
  #steps/nnet/make_denlats.sh --nj 6 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.0833
fi



if [ $stage -le 34 ]; then
  $cuda_cmd $dir/_train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.008 \
    $data_fmllr/train_worn_cleaned $data_fmllr/dev_worn data/lang $ali $ali_dev $dir
fi

if [ $stage -le 35 ]; then
  utils/mkgraph.sh data/lang $dir $dir/graph
  steps/nnet/decode.sh --nj 6  --cmd "$decode_cmd" --acwt 0.0833 --config conf/decode_dnn.config \
    $dir/graph $data_fmllr/eval_worn $dir/decode_eval &
  steps/nnet/decode.sh --nj 6  --cmd "$decode_cmd" --acwt 0.0833 --config conf/decode_dnn.config \
    $dir/graph $data_fmllr/dev_worn $dir/decode_dev &
fi


#if [ $stage -le 17 ]; then
  # chain TDNN
#  local/chain/tuning/run_tdnn_1b.sh --nj ${nj} \
#    --stage $nnet_stage \
#    --train-set ${train_set}_cleaned \
#    --test-sets "$test_sets" \
#    --gmm tri3_cleaned --nnet3-affix _${train_set}_cleaned_rvb
#fi


#if [ $stage -le 19 ]; then
  # final scoring to get the official challenge result
  # please specify both dev and eval set directories so that the search parameters
  # (insertion penalty and language model weight) will be tuned using the dev set
#  local/score_for_submit.sh \
#      --dev exp/chain_${train_set}_cleaned_rvb/tdnn1b_sp/decode_dev_${enhancement}_dereverb_ref \
#      --eval exp/chain_${train_set}_cleaned_rvb/tdnn1b_sp/decode_eval_${enhancement}_dereverb_ref
#fi
