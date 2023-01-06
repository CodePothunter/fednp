#!/bin/bash



## Begin configuration section
stage=0
nj=4
cmd=run.pl



acwt=0.10 # note: only really affects pruning (scoring is on lattices).
beam=13.0
latbeam=8.0
min_active=200
max_active=7000 # limit of active tokens
max_mem=50000000 # approx. limit to memory consumption during minimization in bytes

skip_scoring=false
scoring_opts="--min-lmwt 4 --max-lmwt 15"


splice_opts=
splice_size=
norm_vars=
add_deltas=

## End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: decode.sh [options] <data-dir> <graph-dir> <dnn-dir> <decode-dir>"
   echo " e.g.: decode.sh data/test exp/tri2b/graph exp/dnn_5a exp/dnn_5a/decode"
   echo "main options (for others, see top of script file)"
   echo "  --stage                                  # starts from which stage"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --acwt <acoustic-weight>                 # default 0.1 ... used to get posteriors"
   echo "  --scoring-opts <opts>                    # options to local/score.sh"
   exit 1;
fi

data=$1
graphdir=$2
dnndir=$3
dir=`echo $4 | sed 's:/$::g'` # remove any trailing slash.

srcdir=`dirname $dir`; # assume model directory one level up from decoding directory.
sdata=$data/split$nj;

mkdir -p $dir/log
split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

# Some checks.  Note: we don't need $srcdir/tree but we expect
# it should exist, given the current structure of the scripts.
for f in $graphdir/HCLG.fst $data/feats.scp $dnndir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


export device=cuda0

## Set up the features
echo "$0: feature:cxt_splice(${splice_opts}) time_step_splice(${splice_size}) norm_vars(${norm_vars}) add_deltas(${add_deltas})"
feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
feats="$feats splice-feats  $splice_opts  ark:-  ark:- |"
feats="$feats python  steps_fednp/nnet-forward_sru_fl.py $srcdir/dnn.nnet.pth $srcdir/dnn.nnetD.pth $srcdir/dnn.priors.csv $splice_size $srcdir/learning.json  |"

$cmd JOB=1:$nj $dir/log/decode.JOB.log \
  latgen-faster-mapped --max-active=$max_active --beam=$beam --lattice-beam=$latbeam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt $dnndir/final.mdl $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.JOB.gz"

echo "================+++++++++++++++++++++++++++++++++++++++++++++++++++++++++============================"

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "$0: not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
fi

exit 0;
