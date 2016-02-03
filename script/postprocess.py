#!/usr/bin/env python
# encoding: utf-8
# File Name: postprocess.py
# Author: Jiezhong Qiu
# Create Time: 2016/02/02 20:27
# TODO:

import util
import json
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def predict(feature_file, class_name=None, n_iter=10, test_size=.5):
    X, y = [], []
    with open(feature_file, "rb") as f:
        for line in f:
            data = line.strip().split("\t")
            this_x = [float(x) for x in data[1:]]
            X.append(this_x)
            y.append(int(data[0]))
    X = np.array(X)
    y = np.array(y)
    cross = cross_validation.ShuffleSplit(len(X), n_iter=n_iter, test_size=test_size, random_state=13)
    ma = []
    mi = []
    for train_index, test_index in cross:
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        classif = OneVsRestClassifier(SVC(kernel='rbf', class_weight="auto", verbose=False), n_jobs=10)
        #classif = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", multi_class="ovr", verbose=1)
        classif.fit(X_train, y_train)
        y_pred = classif.predict(X_test)
        macro = f1_score(y_test, y_pred, average='macro')
        micro = f1_score(y_test, y_pred, average='micro')
        print(classification_report(y_test, y_pred, target_names=class_name))
        print(confusion_matrix(y_test, y_pred))
        print "micro F1 = ", micro
        print "macro F1 = ", macro
        ma.append(macro)
        mi.append(micro)
    micro = np.average(mi)
    macro = np.average(ma)
    print "avg micro F1 = ", micro
    print "avg macro F1 = ", macro


def feature(paper_file, action_repre_file, entity_repre_file, action_file, entity_file, feature_file, class_name):
    with open(paper_file, "rb") as f:
        paper = json.load(f)
    action_vector = util.readWord2Vec(action_repre_file, action_file)
    entity_vector = util.readWord2Vec(entity_repre_file, entity_file)
    with open(feature_file, "wb") as f:
        for k, w in paper.iteritems():
            c = w["conference"].strip().split()[0].strip().lower()
            if c not in class_name: continue
            feature = [class_name[c]]
            a_vec = [action_vector[x] for x in w["keywords"] if x in action_vector]
            if len(a_vec) == 0: continue
            a_vec = np.average(np.array(a_vec), axis=0)
            feature += a_vec.tolist()
            e_vec = [entity_vector[x] for x in w["author"] if x in entity_vector]
            if len(e_vec) == 0: continue
            e_vec = np.average(np.array(e_vec), axis=0)
            feature += e_vec.tolist()
            print >> f, "\t".join([str(x) for x in feature])


