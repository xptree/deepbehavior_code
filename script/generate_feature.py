#!/usr/bin/env python
# encoding: utf-8
# File Name: generate_feature.py
# Author: Jiezhong Qiu
# Create Time: 2016/02/02 20:15
# TODO:

from postprocess import feature
import os


def searchPathWeight():
    repre_dir = "/home/jiezhong/embedding/src/deepbehavior_code/result/behavior2vec"
    entity_file = "/home/jiezhong/embedding/data/paper_entity.txt"
    action_file = "/home/jiezhong/embedding/data/paper_action.txt"
    paper_file = "/home/jiezhong/embedding/data/paperv7_pp.json"
    output_dir = "/home/jiezhong/embedding/src/deepbehavior_code/result/feature"
    cList = dict([("aaai", 1), ("icml", 2), ("kdd", 3), ("nips", 4), ("www", 5), ("cikm", 6), ("sigir", 7)])
    neighbor_entity = [i for i in xrange(1, 10)]
    self_entity = [i for i in xrange(1, 10)]
    neigbor_action = [0]
    self_action = [i for i in xrange(1, 10)]

    for ne in neighbor_entity:
        for se in self_entity:
            for na in neigbor_action:
                for sa in self_action:
                    if ne+se+na+sa==10:
                        entity_repre_file = os.path.join(repre_dir, "entity_%d_%d_%d_%d.txt" % (ne, se, na, sa))
                        action_repre_file = os.path.join(repre_dir, "action_%d_%d_%d_%d.txt" % (ne, se, na, sa))
                        feature_file = os.path.join(output_dir, "feature_%d_%d_%d_%d.txt" % (ne, se, na, sa))
                        if os.path.isfile(entity_repre_file) and os.path.isfile(action_repre_file):
                            feature(paper_file, action_repre_file, entity_repre_file, action_file, entity_file,
                                    feature_file, cList)

def searchTimeConst():
    repre_dir = "/home/jiezhong/embedding/src/deepbehavior_code/result/behavior2vec"
    entity_file = "/home/jiezhong/embedding/data/paper_entity.txt"
    action_file = "/home/jiezhong/embedding/data/paper_action.txt"
    paper_file = "/home/jiezhong/embedding/data/paperv7_pp.json"
    output_dir = "/home/jiezhong/embedding/src/deepbehavior_code/result/feature"
    cList = dict([("aaai", 1), ("icml", 2), ("kdd", 3), ("nips", 4), ("www", 5), ("cikm", 6), ("sigir", 7)])

    entity_time_const = [0.5, 1, 2, 3, 4, 5]
    action_time_const = [0.5, 1, 2, 3, 4, 5]
    for i, e_t_c in enumerate(entity_time_const):
        if i!=1: continue
        for j, a_t_c in enumerate(action_time_const):
            if j!=1: continue
            entity_repre_file = os.path.join(repre_dir, "entity_time_const_%d_%d.txt" % (i, j))
            action_repre_file = os.path.join(repre_dir, "action_time_const_%d_%d.txt" % (i, j))
            feature_file = os.path.join(output_dir, "feature_time_const_%d_%d.txt" % (i, j))
            if os.path.isfile(entity_repre_file) and os.path.isfile(action_repre_file):
                feature(paper_file, action_repre_file, entity_repre_file, action_file, entity_file,
                        feature_file, cList)




if __name__ == "__main__":
    #searchPathWeight()
    searchTimeConst()

