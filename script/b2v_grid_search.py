#!/usr/bin/env python
# encoding: utf-8
# File Name: rw_grid_search.py
# Author: Jiezhong Qiu
# Create Time: 2016/02/02 15:13
# TODO:

import os
import subprocess


def searchPathWeight():
    work_dir="/home/jiezhong/embedding/src/deepbehavior_code/behavior2vec"
    train_dir="/home/jiezhong/embedding/src/deepbehavior_code/result/random_walk_hpc/"
    output_dir="/home/jiezhong/embedding/src/deepbehavior_code/result/behavior2vec/"
    script_dir="/home/jiezhong/embedding/src/deepbehavior_code/script/"
    num_threads=32
    num_iter=10

    neighbor_entity = [i for i in xrange(1, 10)]
    self_entity = [i for i in xrange(1, 10)]
    neigbor_action = [0]
    self_action = [i for i in xrange(1, 10)]

    for ne in neighbor_entity:
        for se in self_entity:
            for na in neigbor_action:
                for sa in self_action:
                    if ne+se+na+sa==10:
                        train_file = os.path.join(train_dir, "output_%d_%d_%d_%d.txt" % (ne, se, na, sa))
                        scriput_file = os.path.join(script_dir, "run_b2v.sh")
                        output_entity = os.path.join(output_dir, "entity_%d_%d_%d_%d.txt" % (ne, se, na, sa))
                        output_action = os.path.join(output_dir, "action_%d_%d_%d_%d.txt" % (ne, se, na, sa))
                        args = [scriput_file, work_dir, train_file, output_entity, output_action, \
                                   str(num_threads), str(num_iter)]
                        subprocess.call(args)

def searchTimeConst():
    work_dir="/home/jiezhong/embedding/src/deepbehavior_code/behavior2vec"
    train_dir="/home/jiezhong/embedding/src/deepbehavior_code/result/random_walk_hpc/"
    output_dir="/home/jiezhong/embedding/src/deepbehavior_code/result/behavior2vec/"
    script_dir="/home/jiezhong/embedding/src/deepbehavior_code/script/"
    num_threads=28
    num_iter=100

    entity_time_const = [0.5, 1, 2, 3, 4, 5]
    action_time_const = [0.5, 1, 2, 3, 4, 5]
    for i, e_t_c in enumerate(entity_time_const):
        if i!=1: continue
        for j, a_t_c in enumerate(action_time_const):
            if j!=1: continue
            train_file = os.path.join(train_dir, "output_time_const_%d_%d.txt" % (i, j))
            scriput_file = os.path.join(script_dir, "run_b2v.sh")
            output_entity = os.path.join(output_dir, "entity_time_const_%d_%d.txt" % (i, j))
            output_action = os.path.join(output_dir, "action_time_const_%d_%d.txt" % (i, j))
            args = [scriput_file, work_dir, train_file, output_entity, output_action, \
                        str(num_threads), str(num_iter)]
            subprocess.call(args)

if __name__ == "__main__":
    #searchPathWeight()
    searchTimeConst()
