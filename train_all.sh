#!/usr/bin/env bash

# Simple grid search
logger="train_all.out"
resultsdir="./contrastive_learning_results/"
batch_size=628
echo "STARTING" >$logger
for contrastive_mode in polpol cartcart polcart cartpol; do
  echo " " >>$logger
  for temperature in 0.01 0.02 0.05 0.075 0.1 0.2 0.5 1.0; do
    echo $temperature >>$logger
    dirname="$resultsdir""$contrastive_mode"_clr_temp_"$temperature"
    dirname_nw="$(echo -e "${dirname}" | tr -d '[:space:]')"
    python3 run.py --model_dir="$dirname_nw" --train_batch_size="$batch_size" --temperature="$temperature" --contrastive_mode="$contrastive_mode"
  done
done
