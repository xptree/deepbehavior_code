#!/usr/bin/env python
# encoding: utf-8
# File Name: util.py
# Author: Jiezhong Qiu
# Create Time: 2016/02/02 20:28
# TODO:

def readWord2Vec(repre_file, name_file):
    index = {}
    with open(repre_file, "rb") as f:
        nu = 0
        for line in f:
            nu += 1
            if nu == 1:
                continue
            data = line.strip().split()
            x = [float(item) for item in data[1:]]
            index[int(data[0])] = x
    mapping = {}
    with open(name_file, "rb") as f:
        for line in f:
            content = line.strip().split("\t", 1)
            if int(content[0]) not in index: continue
            mapping[content[1]] = index[int(content[0])]
    return mapping
