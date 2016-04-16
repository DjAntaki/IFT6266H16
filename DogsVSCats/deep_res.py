#!/usr/bin/env python

"""
Only minors changed were maded by me, to ease my use of the code.


Most (nearly all) the rest of the code is from Asojov.
https://github.com/alrojo/lasagne_residual_network/blob/master/Deep_Residual_Network_mnist.py


Lasagne implementation of ILSVRC2015 winner on the mnist dataset.
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

# helper function for projection_b
def ceildiv(a, b):
    return -(-a // b)

def build_cnn(input_var=None, image_shape=(64,64), n=1, num_filters=8):
    # Setting up layers
    conv = lasagne.layers.Conv2DLayer
    #import lasagne.layers.dnn              #Remove if GPU and comment above
    #conv = lasagne.layers.dnn.Conv2DDNNLayer 
 
    nonlin = lasagne.nonlinearities.rectify
    nonlin_layer = lasagne.layers.NonlinearityLayer
    sumlayer = lasagne.layers.ElemwiseSumLayer
    #batchnorm = BatchNormLayer.BatchNormLayer
    batchnorm = lasagne.layers.BatchNormLayer
    expression = lasagne.layers.ExpressionLayer
    pad = lasagne.layers.PadLayer

    # option A for projection as described in paper
    # (should perform slightly worse than B)
    def projection_a(l_inp):
       n_filters = l_inp.output_shape[1]*2
       l = expression(l_inp, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], ceildiv(s[2], 2), ceildiv(s[3], 2)))
       l = pad(l, [n_filters//4,0,0], batch_ndim=1)
       return l

    # option B for projection as described in paper
    def projection_b(l_inp):
        # twice normal channels when projecting!
        n_filters = l_inp.output_shape[1]*2 
        l = conv(l_inp, num_filters=n_filters, filter_size=(1, 1),
                 stride=(2, 2), nonlinearity=None, pad='same', b=None)
        l = batchnorm(l)
        return l

    # helper function to handle filters/strides when increasing dims
    def filters_increase_dims(l, increase_dims):
        in_num_filters = l.output_shape[1]
        if increase_dims:
            first_stride = (2, 2)
            out_num_filters = in_num_filters*2
        else:
            first_stride = (1, 1)
            out_num_filters = in_num_filters
 
        return out_num_filters, first_stride

    # block as described and used in cifar in the original paper:
    # http://arxiv.org/abs/1512.03385
    def res_block_v1(l_inp, nonlinearity=nonlin,
                     increase_dim=False, projection=False):
        # first figure filters/strides
        n_filters, first_stride = filters_increase_dims(l_inp, increase_dim)
        # conv -> BN -> nonlin -> conv -> BN -> sum -> nonlin
        l = conv(l_inp, num_filters=n_filters, filter_size=(3, 3),
                 stride=first_stride, nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=n_filters, filter_size=(3, 3),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        if increase_dim:
            if projection:
                # projection shortcut option B in paper
                p = projection_b(l_inp)
            else:
                # identity shortcut, option A in paper
                p = projection_a(l_inp)
        else:
            p = l_inp
        l = sumlayer([l, p])
        l = nonlin_layer(l, nonlinearity=nonlin)
        return l

    # block as described in second paper on the subject (by same authors):
    # http://arxiv.org/abs/1603.05027
    def res_block_v2(l_inp, nonlinearity=nonlin,
                     increase_dim=False, projection=False):
        # first figure filters/strides
        n_filters, first_stride = filters_increase_dims(l_inp, increase_dim)
        # BN -> nonlin -> conv -> BN -> nonlin -> conv -> sum
        l = batchnorm(l_inp)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=n_filters, filter_size=(3, 3),
                 stride=first_stride, nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=n_filters, filter_size=(3, 3),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        if increase_dim:
            if projection:
                # projection shortcut option B in paper
                p = projection_b(l_inp)
            else:
                # identity shortcut, option A in paper
                p = projection_a(l_inp)
        else:
            p = l_inp
        l = sumlayer([l, p])
        return l

    def bottleneck_block(l_inp, nonlinearity=nonlin,
                         increase_dim=False, projection=False):
        # first figure filters/strides
        n_filters, first_stride = filters_increase_dims(l_inp, increase_dim)
        # conv -> BN -> nonlin -> conv -> BN -> nonlin -> conv -> BN -> sum
        # -> nonlin
        # first make the bottleneck, scale the filters ..!
        scale = 4 # as per bottleneck architecture used in paper
        scaled_filters = n_filters/scale
        l = conv(l_inp, num_filters=scaled_filters, filter_size=(1, 1),
                 stride=first_stride, nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=scaled_filters, filter_size=(3, 3),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=n_filters, filter_size=(1, 1),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        if increase_dim:
            if projection:
                # projection shortcut option b in paper
                p = projection_b(l_inp)
            else:
                # identity shortcut, option a in paper
                p = projection_a(l_inp)
        else:
            p = l_inp
        l = sumlayer([l, p])
        l = nonlin_layer(l, nonlinearity=nonlin)
        return l

    # Bottleneck architecture with more efficiency (the post with Kaiming He's response)
    # https://www.reddit.com/r/MachineLearning/comments/3ywi6x/deep_residual_learning_the_bottleneck/ 
    def bottleneck_block_fast(l_inp, nonlinearity=nonlin,
                     increase_dim=False, projection=False):
        # first figure filters/strides
        n_filters, last_stride = filters_increase_dims(l_inp, increase_dim)
        # conv -> BN -> nonlin -> conv -> BN -> nonlin -> conv -> BN -> sum
        # -> nonlin
        # first make the bottleneck, scale the filters ..!
        scale = 4 # as per bottleneck architecture used in paper
        scaled_filters = n_filters/scale
        l = conv(l_inp, num_filters=scaled_filters, filter_size=(1, 1),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=scaled_filters, filter_size=(3, 3),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=n_filters, filter_size=(1, 1),
                 stride=last_stride, nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        if increase_dim:
            if projection:
                # projection shortcut option b in paper
                p = projection_b(l_inp)
            else:
                # identity shortcut, option a in paper
                p = projection_a(l_inp)
        else:
            p = l_inp
        l = sumlayer([l, p])
        l = nonlin_layer(l, nonlinearity=nonlin)
        return l


    #res_block = res_block_v1       
<<<<<<< HEAD
  #  res_block = res_block_v2
    res_block = lambda x,nonlinearity=nonlin, increase_dim=False : bottleneck_block_fast(x, nonlinearity,
                     increase_dim, projection=True)
=======
   # res_block = res_block_v2
    res_block = lambda x,nonlinearity=nonlin, increase_dim=False :bottleneck_block_fast(x, nonlinearity, increase_dim, projection=True)

>>>>>>> fb6b7dd17e5ab67436c32906f861da3c8ce0642b

    # Stacks the bottlenecks, makes it easy to model size of architecture with int n   
    def blockstack(l, n, nonlinearity=nonlin):
        for _ in range(n):
            l = res_block(l, nonlinearity=nonlin)
        return l

    # Building the network
    l_in = lasagne.layers.InputLayer(shape=(None, 3, image_shape[0], image_shape[1]),
                                        input_var=input_var)
    # First layer! just a plain convLayer
    l1 = conv(l_in, num_filters=num_filters, stride=(1, 1),
              filter_size=(3, 3), nonlinearity=None, pad='same')
    l1 = batchnorm(l1)
    l1 = nonlin_layer(l1, nonlinearity=nonlin)

    # Stacking bottlenecks and increasing dims! (while reducing shape size)
    l1_bs = blockstack(l1, n=n)
    l1_id = res_block(l1_bs, increase_dim=True)

    l2_bs = blockstack(l1_id, n=n)
    l2_id = res_block(l2_bs, increase_dim=True)

    l3_bs = blockstack(l2_id, n=n)
    l3_id = res_block(l3_bs, increase_dim=True)

    # And, finally, the output layer:
    network = lasagne.layers.DenseLayer(
            l3_id,
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network
