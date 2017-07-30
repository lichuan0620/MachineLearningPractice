#!/usr/bin/env python

"""2016W-CS178: Homework 3, Problem1"""


import numpy as np
import matplotlib.pyplot as plt
import mltools as ml


iris = np.genfromtxt("data/iris.txt")
features = iris[:, 0:2]
targets = iris[:, -1]
features, targets = ml.shuffleData(features, targets)
features, _ = ml.transforms.rescale(features)
# sub1: class 0 and class 1
features_sub1 = features[targets < 2, :]
targets_sub1 = targets[targets < 2]
# sub2: class 1 and class 2
features_sub2 = features[targets > 0, :]
targets_sub2 = targets[targets > 0]
# features by classes
features_c0 = features[targets == 0, :]
features_c1 = features[targets == 1, :]
features_c2 = features[targets == 2, :]


plt.figure(1, (15, 7))
plt.subplot(121)
plt.scatter(features_c0[:, 0], features_c0[:, 1], color='r', label='class 0')
plt.scatter(features_c1[:, 0], features_c1[:, 1], color='b', label='class 1')
plt.title('Class 0 vs Class 1')
plt.xlabel('class 0')
plt.ylabel('class 1')
plt.grid(1)
plt.legend(loc='upper center', ncol=2)
plt.subplot(122)
plt.scatter(features_c1[:, 0], features_c1[:, 1], color='r', label='class 1')
plt.scatter(features_c2[:, 0], features_c2[:, 1], color='b', label='class 2')
plt.title('Class 1 vs Class 2')
plt.xlabel('class 1')
plt.ylabel('class 2')
plt.grid(1)
plt.legend(loc='upper center', ncol=2)
plt.show()

plt.figure(2, (15, 7))
# empty learner for testing #
learner = ml.logistic2.logisticClassify2()
learner.theta = np.mat([[.5], [1], [-.25]])
#############################
plt.subplot(121)
learner.plotBoundary(features_sub1, targets_sub1)
plt.legend()
plt.subplot(122)
learner.plotBoundary(features_sub2, targets_sub2)
plt.legend()
plt.show()
