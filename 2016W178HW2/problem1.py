#!/usr/bin/env python

"""2016W-CS178: Homework 2, Problem1"""

import numpy as np
import mltools as ml
import matplotlib.pyplot as plt

data = np.genfromtxt("data/curve80.txt")
features = (data[:, 0])[:, np.newaxis]
targets = data[:, 1]
train_features, test_features, train_targets, test_targets = ml.splitData(features, targets, 0.75)
train_data_point = train_features.shape[0]
test_data_points = test_features.shape[0]
train_error = []
test_error = []
degrees = (1, 3, 5, 7, 10, 18)

plt.figure(1, (17, 7))
plt.subplot(1, 2, 1)
plt.scatter(train_features, train_targets, color='b', label='training data')
plt.scatter(test_features, test_targets, color='r', label='test data')

for degree in degrees:
    # prepare data
    poly_train_features, params = ml.transforms.rescale(ml.transforms.fpoly(train_features, degree, 0))
    poly_test_features = ml.transforms.rescale(ml.transforms.fpoly(test_features, degree, 0), params)[0]

    # learn and predict
    learner = ml.linear.linearRegress(poly_train_features, train_targets)
    predicted_train_targets = learner.predict(poly_train_features).flatten()
    predicted_test_targets = learner.predict(poly_test_features).flatten()

    # calculate error
    train_error.append(np.sum(np.power(predicted_train_targets - train_targets, 2))/float(train_data_point))
    test_error.append(np.sum(np.power(predicted_test_targets - test_targets, 2))/float(test_data_points))

    # plot prediction function
    x = (np.linspace(np.amin(train_features), np.amax(train_features), 200))[:, np.newaxis]
    y = learner.predict(ml.transforms.rescale(ml.transforms.fpoly(x, degree, 0), params)[0])
    plt.plot(x, y, label='Degree %d' % degree)

plt.title('training results with different degrees', fontsize=18)
plt.xlabel('feature', fontsize=14)
plt.ylabel('target', fontsize=14)
plt.grid(1)
plt.axis([np.amin(features)-2, np.amax(features)+2, np.amin(targets)-2, np.amax(targets)+2])
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.title('MSE vs degree', fontsize=18)
plt.semilogy(degrees, train_error, label='Train set MSE')
plt.semilogy(degrees, test_error, label='Test set MSE')
plt.xlabel('degree', fontsize=14)
plt.ylabel('error', fontsize=14)
plt.grid(1)
plt.legend(loc='upper center', ncol=2)

plt.show()
