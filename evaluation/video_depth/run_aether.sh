#!/bin/bash

export PYTHONPATH="./"

set -e

workdir='.'
model_name='aether'
datasets=('sintel' 'kitti' 'bonn')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_${model_name}"
    echo "$output_dir"
    accelerate launch --main_process_port $RANDOM --num_processes 1 \
        evaluation/video_depth/launch_aether.py \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
done


for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_${model_name}"
    echo "$output_dir"
    python evaluation/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale"
done
