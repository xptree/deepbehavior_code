#!/bin/bash

set -e 
#entity_file="/home/jiezhong/embedding/data/paper_edge.txt"
#action_file="/home/jiezhong/embedding/data/paper_action_edge.txt"
#behavior_file="/home/jiezhong/embedding/data/paper_history.txt"

work_dir=$1
entity_file=$2
action_file=$3
behavior_file=$4
neighbor_entity=$5
self_entity=$6
neighbor_action=$7
self_action=$8
num_threads=$9
output_file=${10}
entity_time_const=${11}
action_time_const=${12}


$work_dir/main -entity_file $entity_file -behavior_file $behavior_file \
	-num_walks_entity 10 -num_walks_action 200 -walk_length 40 -num_threads $num_threads \
	-entity_time_const $entity_time_const -action_time_const $action_time_const -entity_relation_const 1 -action_relation_const 1 \
	-neighbor_entity $neighbor_entity -self_entity $self_entity -neighbor_action $neighbor_action -self_action $self_action \
	-output_file $output_file
