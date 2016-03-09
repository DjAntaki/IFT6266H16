#!/usr/bin/env python

"""
FROM : alrojo
https://github.com/alrojo/lasagne_residual_network/blob/master/Deep_Residual_Network_mnist.py

I modified it very slightly for it to work on DogsVsCats.

Lasagne implementation of ILSVRC2015
Deep Residual Learning for Image Recognition
http://arxiv.org/abs/1512.03385
"""

from __future__ import print_function

import sys
import os
import time
import string

import numpy as np
import theano
import theano.tensor as T

import lasagne
import BatchNormLayer
sys.setrecursionlimit(10000)


# ##################### Build the neural network model #######################

def build_cnn(input_var=None, image_size=(150,150), n=1, num_filters=64, output_size=2):
    # Setting up layers
    conv = lasagne.layers.Conv2DLayer
    #import lasagne.layers.dnn
#    conv = lasagne.layers.dnn.Conv2DDNNLayer # cuDNN
    nonlinearity = lasagne.nonlinearities.rectify
    sumlayer = lasagne.layers.ElemwiseSumLayer
#    scaleandshiftlayer = parmesan.layers.ScaleAndShiftLayer
#    normalizelayer = parmesan.layers.NormalizeLayer
    batchnorm = BatchNormLayer.batch_norm
    # Conv layers must have batchnormalization and
    # Micrsoft PReLU paper style init(might have the wrong one!!)
    def convLayer(l, num_filters, filter_size=(1, 1), stride=(1, 1),
                  nonlinearity=nonlinearity, pad='same', W=lasagne.init.HeNormal(gain='relu')):
        l = conv(l, num_filters=num_filters, filter_size=filter_size,
            stride=stride, nonlinearity=nonlinearity,
            pad=pad, W=W)
        # Notice that the batch_norm layer reallocated the nonlinearity form the conv
        l = batchnorm(l)
        return l
    
    # Bottleneck architecture as descriped in paper
    def bottleneckDeep(l, num_filters, stride=(1, 1), nonlinearity=nonlinearity):
        l = convLayer(
            l, num_filters=num_filters, stride=stride, nonlinearity=nonlinearity)
        l = convLayer(
            l, num_filters=num_filters, filter_size=(3, 3), nonlinearity=nonlinearity)
        l = convLayer(
            l, num_filters=num_filters*4, nonlinearity=nonlinearity)
        return l
    # Bottleneck architecture with more efficiency (the post with Kaiming he's response)
    # https://www.reddit.com/r/MachineLearning/comments/3ywi6x/deep_residual_learning_the_bottleneck/
    def bottleneckDeep2(l, num_filters, stride=(1, 1), nonlinearity=nonlinearity):
        l = convLayer(
            l, num_filters=num_filters, nonlinearity=nonlinearity)
        l = convLayer(
            l, num_filters=num_filters, filter_size=(3, 3), stride=stride, nonlinearity=nonlinearity)
        l = convLayer(
            l, num_filters=num_filters*4, nonlinearity=nonlinearity)
        return l
    # The "simple" residual block architecture
    def bottleneckShallow(l, num_filters, stride=(1, 1), nonlinearity=nonlinearity):
        l = convLayer(
            l, num_filters=num_filters*4, filter_size=(3, 3), stride=stride, nonlinearity=nonlinearity)
        l = convLayer(
            l, num_filters=num_filters*4, filter_size=(3, 3), nonlinearity=nonlinearity)
        return l
        
    bottleneck = bottleneckShallow

    # Simply stacks the bottlenecks, makes it easy to model size of architecture with int n   
    def bottlestack(l, n, num_filters):
        for _ in range(n):
            l = sumlayer([bottleneck(l, num_filters=num_filters), l])
        return l

    # Building the network
    l_in = lasagne.layers.InputLayer(shape=(None, 3, image_size[0], image_size[1]),
                                        input_var=input_var)
    # First layer just a plain convLayer
    l1 = convLayer(
	    l_in, num_filters=num_filters*4, filter_size=(3, 3)) # Filters multiplied by 4 as bottleneck returns such size

    # Stacking bottlenecks and making residuals!

    l1_bottlestack = bottlestack(l1, n=n-1, num_filters=num_filters) #Using the -1 to make it fit with size of the others
    l1_residual = convLayer(l1_bottlestack, num_filters=num_filters*4*2, stride=(2, 2), nonlinearity=None) #Multiplying by 2 because of feature reduction by 2

    l2 = sumlayer([bottleneck(l1_bottlestack, num_filters=num_filters*2, stride=(2, 2)), l1_residual])
    l2_bottlestack = bottlestack(l2, n=n, num_filters=num_filters*2)
    l2_residual = convLayer(l2_bottlestack, num_filters=num_filters*2*2*4, stride=(2, 2), nonlinearity=None)# again, this is now the second reduciton in features

    l3 = sumlayer([bottleneck(l2_bottlestack, num_filters=num_filters*2*2, stride=(2, 2)), l2_residual])
    l3_bottlestack = bottlestack(l3, n=n, num_filters=num_filters*2*2)

    # And, finally, the 10-unit output layer:
    network = lasagne.layers.DenseLayer(
            l3_bottlestack,
            num_units=output_size,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network
