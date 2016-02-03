#!/bin/bash

set -e

work_dir=$1
train_file=$2
output_entity=$3
output_action=$4
num_threads=$5
num_iter=$6

#/home/jiezhong/embedding/src/deepBehavior/random_walk_hpc
#./word2vec -train /home/jiezhong/embedding/src/deepBehavior/random_walk_hpc/output.txt -output_entity entity.txt -output_action action.txt -size 100 -window 5 -negative 5 -hs 1 -cbow 0 -iter 10 -debug 2 -threads 32 -min-count 1
#./deepbehavior -train /home/jiezhong/embedding/src/deepBehavior/random_walk_hpc/output.txt -output_entity entity_.txt -output_action action_.txt -size 30 -window 5 -negative 5 -hs 1 -cbow 0 -iter 3 -debug 2 -threads 32 -min-count 1
$work_dir/main -train $train_file -output_entity $output_entity -output_action $output_action \
    -size 30 -window 5 -negative 5 -hs 1 -cbow 0 -iter $num_iter -debug 2 -threads $num_threads -min-count 1
