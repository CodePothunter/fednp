#!/bin/bash
for i in `seq 4`; do echo $i; sort -n /tmp/sbbb.txt `cat exp/sru_fl_12_1_0.0003/decode_$i/wer_* | grep WER | cut -d ' ' -f 2 > /tmp/sbbb.txt` | head -n 1; done
