#!/usr/bin/env python

# by    Chuan Li
# since 08/08/2017

"""
A simple MNIST classifier made with TensorFlow

This is a study project made to explore some basic features of TensorFlow.
The project was written following the instructions here:
    https://www.tensorflow.org/get_started/mnist/pros
"""

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MINST_data', one_hot=True)
