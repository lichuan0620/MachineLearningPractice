#!/usr/bin/env python

"""2016W-CS178: Homework 1, Problem1"""

import numpy
import matplotlib.pyplot as plt

iris = numpy.genfromtxt("data/iris.txt", delimiter=None)
Y = iris[:, -1]
X = iris[:, 0:-1]

# problem 1(a)
features = X.shape[1]
dpoints = X.shape[0]
print('features:\t\t%d' % features)
print('data points:\t%d' % dpoints)

# problem 1(b)
plt.figure(1, (12,9))

for f in range(0,features):
    plt.subplot(2, 2, f+1) # when features == 4
    plt.hist(X[:, f], dpoints/4, facecolor='g')
    plt.xlabel('value')
    plt.ylabel('count')
    plt.title('iris: data values %d' % f)
    plt.grid(1)

plt.show()
plt.close(1)

# problem 1(c)
meanX = numpy.mean(X, axis=0)
print('mean:\t\t\t' + str(meanX))

# problem 1(d)
varX = numpy.var(X, axis=0)
stdX = numpy.std(X, axis=0)
print('variance:\t\t' + str(varX))
print('std deviation:\t' + str(stdX))

# problem 1(e)
normX = X

for row in range(0, dpoints):
    normX[row] = (normX[row] - meanX)/stdX

# problem 1(f)
plt.figure(2, (15, 5))

for i in range(1,4):
    plt.subplot(1,3,i)
    plt.scatter(normX[:, 0], normX[:, i], c=Y)
    plt.xlabel('feature 1')
    plt.ylabel('feature %d' % (i+1))
    plt.grid(1)

plt.show()
plt.close(2)
