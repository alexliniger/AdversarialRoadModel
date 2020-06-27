#!/usr/bin/env bash

min=$(date +%Y%m%d_%H%M%S)

save_dir="Data/$min"
mkdir -p $save_dir

code_dir="$save_dir/code"
rm -rf $code_dir
mkdir -p $code_dir
rsync -av --progress . $code_dir --exclude Data --exclude .idea/

result_dir="$save_dir/results"
rm -rf $result_dir
mkdir -p $result_dir

python train.py --path $result_dir