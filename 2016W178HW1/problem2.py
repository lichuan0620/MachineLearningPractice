#!/usr/bin/env python

"""2016W-CS178: Homework 1, Problem2"""

import numpy

import mltools

iris = numpy.genfromtxt("data/iris.txt", delimiter=None)
Y = iris[:, -1]
X = iris[:, 0:-1]
X, Y = mltools.shuffleData(X, Y)
trainX, testX, trainY, testY = mltools.splitData(X, Y, 0.75)