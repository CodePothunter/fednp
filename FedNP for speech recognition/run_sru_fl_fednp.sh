#!/bin/bash


set -e

nj=8
. cmd.sh
. path.sh

N=16
turns=12
epoch_per_turn=1


## Configurable directories

# train: the federated setting root director, should contain 16 subdirectors 
# each for one party.
train=data-fbank/train_worn_cleaned/split$N

dev=data-fbank/dev_worn
test=data-fbank/eval_worn



lang=data/lang
gmm=exp/tri6a_dnn
exp=exp/sru_fedep_${N}_${turns}_${epoch_per_turn}_${batchSize}
lm=tgpr_5k

export DEVICE=cuda
export CUDA_VISIBLE_DEVICES=3

## tune learning rate
## Train
for lr in 0.0003; do
    python steps_fednp/train_sru_fl.py $dev ${gmm}_ali_dev $train ${gmm}_ali $gmm ${exp}_${lr} $lr $N $turns $epoch_per_turn
    echo "Done deep learning"
    export CUDA_VISIBLE_DEVICES=6
    # exit 1

    ## Make graph
    [ -f $gmm/graph/HCLG.fst ] || utils/mkgraph.sh $lang $gmm ${gmm}/graph

    ## Decode
    echo "tune acoustic scale"
    for ac in 0.08333  ; do
    [ -f ${exp}_${lr}_${lmada}/decode.done ] || bash steps_fednp/decode_sru_fl.sh --nj $nj --acwt $ac --scoring-opts "--min-lmwt 4 --max-lmwt 15"  \
     --add-deltas "true" --norm-vars "true" --splice-size "20" --splice-opts "--left-context=0 --right-context=4"  \
     $test $gmm/graph ${exp}_${lr} ${exp}_${lr}/decode_$ac
    done 
done
sort -n /tmp/sbbb.txt `cat ${exp}_${lr}/decode_*/wer_* | grep WER | cut -d ' ' -f 2 > /tmp/sbbb.txt` | head -n 1


    
