#!/usr/bin/env python

"""2016W-CS178: Homework 1, Problem3"""

import numpy

X = numpy.array([
    [0, 0, 1, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 1, 1, 1]
])
Y = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0])
x1 = numpy.array([0, 0, 0, 0, 0])
x2 = numpy.array([1, 1, 0, 1, 0])


def predict(x, p):
    """predict outcome y based on feature list x and p(x|y) list p"""
    if len(x) is not len(p):
        print('The feature array (size: %d) and the probability array (size: %d) must have the same size')
    else:
        to_return = 1
        for f in range(0, len(x)):
            to_return *= p[f] if x[f] else 1 - p[f]
        return to_return


# problem 3(a)
tNonSpam = numpy.sum(Y)     # number of non-spam emails
pXgY = []                   # p(xi|y) or chance of feature xi if the email is not spam

for i in range(0, X.shape[1]):
    pXgY.append(sum(X[:, i]*Y)/float(tNonSpam))

print('p(y|xi): ' + str(['%.04f' % i for i in pXgY]))

# problem 3(c)
print('p(y|'+str(x1)+'): ' + '%.04f' % predict(x1, pXgY))
print('p(y|'+str(x2)+'): ' + '%.04f' % predict(x2, pXgY))
