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

learner = ml.logistic2.logisticClassify2(features_sub1, targets_sub1, plot=1)
print 'subset 1 error: %f' % learner.err(features_sub1, targets_sub1)
learner2 = ml.logistic2.logisticClassify2(features_sub2, targets_sub2, plot=2)
print 'subset 2 error: %f' % learner2.err(features_sub2, targets_sub2)
plt.figure(3, figsize=(15, 7))
plt.subplot(121)
learner.plotBoundary(features_sub1, targets_sub1)
plt.legend()
plt.subplot(122)
learner2.plotBoundary(features_sub2, targets_sub2)
plt.legend()
plt.show()
