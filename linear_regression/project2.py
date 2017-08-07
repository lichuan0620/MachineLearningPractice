#!/usr/bin/env python

"""2016W-CS178: Homework 2, Problem3"""

# TODO: make this generic enough and rewrite as a class

import numpy as np
import mltools as ml
import matplotlib.pyplot as plt


report = 1


def print_report(to_print):
    if report:
        print to_print


# data
print_report('loading data: training feature set')
features_train  = np.genfromtxt('data/kaggle.X1.train.txt', delimiter=',') # training features
print_report('loading data: training target set')
targets_train   = np.genfromtxt('data/kaggle.Y.train.txt', delimiter=',')  # training targets
print_report('loading data: testing feature set')
features_test   = np.genfromtxt('data/kaggle.X1.test.txt', delimiter=',')  # testing targets

# --- visualization of the data by feature  ---
# for i in range(0, features_train.shape[0]):
#     plt.figure(figsize=(12,10))
#     plt.scatter(ml.transforms.rescale(features_train[:, i])[0], targets_train)
#     plt.title('feature %d' % i)
#     plt.show()
# ---------------------------------------------

# constants
train_dp    = features_train.shape[0] # number of data points of the training data
test_dp     = features_test.shape[0] # number of data points of the testing data
cv_k        = 5 # k value for k-fold cross validate
degrees     = range(2, 15, 3) # polynomial degrees on which the training/testing will take place


# train and test

error_cv    = [] # MSE on different polynomial degrees obtained by k-fold cross validation
error_test  = [] # MSE on different polynomial degrees obtained by testing
print_report('training starts...')
for degree in degrees:
    print_report('polynomial degree %d' % degree)
    # k-fold cross validation
    c_error_d = [] # MSE on each fold
    for k in range(0, cv_k):
        print_report('k-fold cross validation, fold %d' % k)
        x_train, x_test, y_train, y_test = ml.crossValidate(features_train, targets_train, cv_k, k)
        x_train_, params = ml.transforms.rescale(ml.transforms.fpoly(x_train, degree, 0))
        x_test = ml.transforms.rescale(ml.transforms.fpoly(x_test, degree, 0), params)[0]
        learner = ml.linear.linearRegress(x_train, y_train)
        y_predicted = learner.predict(y_test)

