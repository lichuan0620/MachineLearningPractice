#!/usr/bin/env python

"""2016W-CS178: Homework 1, Problem2"""

import numpy
import matplotlib.pyplot as plt
import mltools

iris = numpy.genfromtxt("data/iris.txt")
Y = iris[:, -1]
X = iris[:, 0:2] # feature 1 & 2
X, Y = mltools.shuffleData(X, Y)
trainX, testX, trainY, testY = mltools.splitData(X, Y, 0.75)

# problem 2(a)
plt.figure(1, (12, 9))

for i, k in enumerate([1, 5, 10, 50]):
    learner = mltools.knn.knnClassify()
    learner.train(trainX, trainY, k)
    plt.subplot(2, 2, i+1)
    mltools.plot_classify_2d(learner, trainX, trainY)
    plt.grid(1)
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.title('Iris KNN: Feature 1 & 2, K = %d' % k)

plt.show()
plt.close(1)

# problem 2(b)
K = [1, 2, 5, 10, 50, 100, 200]
errTrain = []
errTest = []
lenTestY = testY.shape[0]
lenTrainY = trainY.shape[0]

for i, k in enumerate(K):
    learner = mltools.knn.knnClassify()
    learner.train(trainX, trainY, k)
    errTrain.append(1 - float(numpy.sum(learner.predict(trainX) == trainY))/float(lenTrainY))
    errTest.append(1 - float(numpy.sum(learner.predict(testX) == testY))/float(lenTestY))

plt.figure(2, (10, 6))
plt.semilogx(K, errTrain, 'rh-', label="train error")
plt.semilogx(K, errTest, 'gH-', label="test error")
plt.legend()
plt.xlabel('K')
plt.ylabel('%error')
plt.title('Iris KNN: percentage error relative to K')
plt.grid(1)
plt.show()
plt.close(2)
