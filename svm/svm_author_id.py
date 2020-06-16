#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
base_path = "C:/Users/Trent.Park/Projects/udacity/ud120-projects"
sys.path.append(base_path + "/tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Experiment on how training time and accuracy differ with a different size training set
# This significantly reduces training time
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

# Trying alternative C parameters
# It turns out that C=10000, using the RBF kernel, gives the best result
C_param_list = [10, 100, 1000, 10000]

for C in C_param_list:

    clf = SVC(kernel='rbf', C=C)

    t0 = time()
    clf.fit(features_train, labels_train)
    print "\nThe training time was: {0}s\n".format(round(time() - t0))

    t0 = time()
    labels_pred = clf.predict(features_test)
    print "\nThe simulation time was: {0}s\n".format(round(time() - t0))

    print "\nThe accuracy of the SVC is: {0}, with C parameter: {1}\n" \
            .format(accuracy_score(labels_test, labels_pred), C)

#########################################################


