#!/usr/bin/env python
# encoding: utf-8
# File Name: rw_grid_search.py
# Author: Jiezhong Qiu
# Create Time: 2016/02/02 15:13
# TODO:

import os
import subprocess

def searchPathWeight():
    work_dir="/home/shawn/lab/Tencent/dnf/deepbehavior_code/random_walk_hpc"
    action_file="/home/shawn/lab/Tencent/dnf/deepbehavior_code/data/output_items_link.txt"
    entity_file="/home/shawn/lab/Tencent/dnf/deepbehavior_code/data/user_link.txt_tab.0"
    behavior_file="/home/shawn/lab/Tencent/dnf/deepbehavior_code/data/log_reduced_ord.txt"
    output_dir="/home/shawn/lab/Tencent/dnf/deepbehavior_code/result/random_walk_hpc"
    script_dir="/home/shawn/lab/Tencent/dnf/deepbehavior_code/script/"
    num_threads=60

    neighbor_entity = [i for i in xrange(1, 10)]
    neighbor_entity = [5]
    self_entity = [i for i in xrange(1, 10)]
    self_entity = [1]
    neigbor_action = [1]
    self_action = [i for i in xrange(1, 10)]
    self_action = [3]
    for ne in neighbor_entity:
        for se in self_entity:
            for na in neigbor_action:
                for sa in self_action:
                    if ne+se+na+sa==10:
                        output_file = os.path.join(output_dir, "output_%d_%d_%d_%d.txt" % (ne, se, na, sa))
                        scriput_file = os.path.join(script_dir, "run_rw.sh")
                        args = [scriput_file, work_dir, entity_file, action_file, behavior_file, \
                                    str(ne/10.), str(se/10.), str(na/10.), str(sa/10.), str(num_threads), output_file, "1", "1"]
                        print args
                        subprocess.call(args)

def searchTimeConst():
    work_dir="/home/jiezhong/embedding/src/deepbehavior_code/random_walk_hpc"
    entity_file="/home/jiezhong/embedding/data/paper_edge.txt"
    action_file="/home/jiezhong/embedding/data/paper_action_edge.txt"
    behavior_file="/home/jiezhong/embedding/data/paper_history.txt"
    output_dir="/home/jiezhong/embedding/src/deepbehavior_code/result/random_walk_hpc"
    script_dir="/home/jiezhong/embedding/src/deepbehavior_code/script/"
    num_threads=28
    ne = 6
    se = 3
    na = 0
    sa = 1

    entity_time_const = [0.5, 1, 2, 3, 4, 5]
    action_time_const = [0.5, 1, 2, 3, 4, 5]
    for i, e_t_c in enumerate(entity_time_const):
        if e_t_c != 1: continue
        for j, a_t_c in enumerate(action_time_const):
            if a_t_c != 1: continue
            output_file = os.path.join(output_dir, "output_time_const_%d_%d.txt" % (i, j))
            scriput_file = os.path.join(script_dir, "run_rw.sh")
            args = [scriput_file, work_dir, entity_file, action_file, behavior_file, \
                        str(ne/10.), str(se/10.), str(na/10.), str(sa/10.), str(num_threads), output_file, str(e_t_c), str(a_t_c)]
            subprocess.call(args)


if __name__ == "__main__":
    searchPathWeight()
    #searchTimeConst()
