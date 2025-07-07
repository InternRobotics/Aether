#!/bin/bash

export PYTHONPATH="./"

set -e

workdir='.'
model_name='aether'
# datasets=('tum' 'sintel' 'scannet')
datasets=('sintel' 'scannet')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/rel_pose/${data}_${model_name}"
    echo "$output_dir"
    accelerate launch --main_process_port $RANDOM --num_processes 1 \
        evaluation/rel_pose/launch_aether.py \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
done
