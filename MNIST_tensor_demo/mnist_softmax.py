#!/usr/bin/env python

# by    Chuan Li
# since 08/08/2017

"""
A simple MNIST classifier made with TensorFlow

(MNIST is a dataset of some (28px x 28px) images of hand written digits. The goal of this classifier
is to recognize these digits by learning from the training set. It uses a softmax (or multinomial
logistic) regression model.

This is a study project made to explore some basic features of TensorFlow. The project was written
following the instructions here:
https://www.tensorflow.org/get_started/mnist/pros
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# Absolute constants
kFeatureCount   = 784   # 28x28 pixels, a total of 784 features
kClassCount     = 10    # 0 to 9, a total of 10 recognizable classes

# Tunable constants:
kTrainingLimit  = 1000
kBatchSize      = 100
kStepConstant   = 0.5


print('Loading data...')
# This statement download the MNIST dataset.
mnist = input_data.read_data_sets('MINST_data', one_hot=True)


print('Creating model...')
y_ = tf.placeholder(tf.float32, shape=[None, kClassCount])   # test results

x = tf.placeholder(tf.float32, shape=[None, kFeatureCount])  # features
W = tf.Variable(tf.zeros([kFeatureCount, kClassCount]))      # weights
b = tf.Variable(tf.zeros([kClassCount]))                     # bias

# y: the (softmax regression) model

# evidence_i = sum_over_j(matrix_multiply(weight_ij, input_j) + bias_i)
#       where i is the index of the data point and j is the index of the class
# targets = softmax(evidence) = normalize(exp(evidence))
#
# For more info on softmax regression, see:
#       http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
y = tf.matmul(x, W) + b

# cross_entropy: the cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# train_step: the gradient
train_step = tf.train.GradientDescentOptimizer(kStepConstant).minimize(cross_entropy)

# correct_prediction: a list of boolean recording the correctness of the predictions
correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
# accuracy: the average of correct_prediction (e.g. [1, 0, 1, 1] = (1 + 1 + 1)/4 = 0.75)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


print('Creating a session...')
# A TensorFlow session is a connection to TensorFlow's C++ backend.
sess = tf.Session()
# Note that, because switching between the Python script and C++ backend for every operation
# is not efficient, TensorFlow adopts the concept of computation graph to load the bulk of the
# operations into a session at once. The tensors (or, basically, the variables) above make up
# a computation graph (or a group of operations represented by the tensors such as the ones
# above). The computation graph is completed before the session is started and the tensors are
# initialized by the statement below.
sess.run(tf.global_variables_initializer())

print('Training starts...')
for i in range(kTrainingLimit):
    batch = mnist.train.next_batch(kBatchSize)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
    train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1]})
    # uncomment to see the accuracy at step i:
    # print('Accuracy @ step %d: %g' % (i, sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1]})))

print('Final accuracy: %g' % sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1]}))
# TODO: improve accuracy (currently the model yields an accuracy of ~93%)
