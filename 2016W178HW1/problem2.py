#!/usr/bin/env python

"""2016W-CS178: Homework 1, Problem2"""

import numpy
import matplotlib.pyplot as plt
import mltools

iris = numpy.genfromtxt("data/iris.txt", delimiter=None)
Y = iris[:, -1]
X = iris[:, 0:-1]
X, Y = mltools.shuffleData(X, Y)
trainX, testX, trainY, testY = mltools.splitData(X, Y, 0.75)

# problem 2(a)
plt.figure(1, (12, 9))

for i, k in enumerate([1, 5, 10, 50]):
    learner = mltools.knn.knnClassify()
    learner.train(trainX[:, 0:2], trainY, k)
    plt.subplot(2, 2, i+1)
    mltools.plotClassify2D(learner, trainX[:, 0:2], trainY)
    plt.grid(1)
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.title('Iris KNN: Feature 1 & 2, K = %d' % k)

plt.show()
plt.close(1)
