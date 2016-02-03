#!/usr/bin/env python
# encoding: utf-8
# File Name: prediction.py
# Author: Jiezhong Qiu
# Create Time: 2016/01/15 19:22
# TODO:

import os
from postprocess import predict

def searchPathWeight():
    feature_dir = "/home/jiezhong/embedding/src/deepbehavior_code/result/feature"
    cList = [("aaai", 1), ("icml", 2), ("kdd", 3), ("nips", 4), ("www", 5), ("cikm", 6), ("sigir", 7)]
    class_name = [item[0] for item in cList]
    n_iter = 10
    test_size = .5
    neighbor_entity = [i for i in xrange(1, 10)]
    neighbor_entity = [4, 5, 6, 7, 8]
    self_entity = [i for i in xrange(1, 10)]
    neigbor_action = [0]
    self_action = [i for i in xrange(1, 10)]

    # deepwalk
    #feature_file = os.path.join(feature_dir, "deepwalk2.txt")
    #predict(feature_file, class_name, n_iter, test_size)

    for ne in neighbor_entity:
        for se in self_entity:
            for na in neigbor_action:
                for sa in self_action:
                    if ne+se+na+sa==10:
                        feature_file = os.path.join(feature_dir, "feature_%d_%d_%d_%d.txt" % (ne, se, na, sa))
                        if os.path.isfile(feature_file):
                            print "********** %s **********" % (feature_file,)
                            predict(feature_file, class_name, n_iter, test_size)
                            print "********** %s **********" % (feature_file,)


def searchTimeConst():
    feature_dir = "/home/jiezhong/embedding/src/deepbehavior_code/result/feature"
    cList = [("aaai", 1), ("icml", 2), ("kdd", 3), ("nips", 4), ("www", 5), ("cikm", 6), ("sigir", 7)]
    class_name = [item[0] for item in cList]
    n_iter = 20
    test_size = .25

    entity_time_const = [0.5, 1, 2, 3, 4, 5]
    action_time_const = [0.5, 1, 2, 3, 4, 5]


    # deepwalk
    feature_file = os.path.join(feature_dir, "deepwalk2.txt")
    predict(feature_file, class_name, n_iter, test_size)
    exit()
    for i, e_t_c in enumerate(entity_time_const):
        if i!=1: continue
        for j, a_t_c in enumerate(action_time_const):
            if j!=1: continue
            feature_file = os.path.join(feature_dir, "feature_time_const_%d_%d.txt" % (i ,j))
            if os.path.isfile(feature_file):
                print "********** %s **********" % (feature_file,)
                predict(feature_file, class_name, n_iter, test_size)
                print "********** %s **********" % (feature_file,)



if __name__ == "__main__":
    searchTimeConst()
