#!/usr/bin/env python
# -*- coding: utf-8 -*-

#A couples of functions to build or assist in building a model.
# DISCLAIMER : not everything works in this file


import theano
from theano import tensor
import numpy as np
from blocks.bricks.recurrent import BaseRecurrent, LSTM, Bidirectional, GatedRecurrent, SimpleRecurrent
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant, Uniform, Identity
from blocks.bricks import Tanh, NDimensionalSoftmax, Linear

from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, Scale
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring

#--
# Functions used during net instanciation.
#---
def testing_init(brick):
    brick.weights_init = Identity()
    brick.biases_init = Constant(0)
    brick.initialize()

def default_init(brick):
    brick.weights_init = Uniform(width=0.08)
    brick.biases_init = Constant(0)
    brick.initialize()

def add_lstm(input_dim, input_var):
    linear = Linear(input_dim=input_dim,output_dim=input_dim*4,name="linear_layer")
    lstm = LSTM(dim=input_dim, name="lstm_layer")

    testing_init(linear)
    #linear.initialize()
    default_init(lstm)

    h = linear.apply(input_var)
    return lstm.apply(h)

def add_softmax_layer(input_var, input_size, output_size):
    """makes a Linear layer   """
    l = Linear(name="linear_before_softmax", input_dim=input_size, output_dim=output_size)  
    default_init(l)
    return l.apply(input_var)
    
#Some models.

def getLSTMstack(input_dim, input_var, depth):
    
    next_input = input_var
    for i in range(depth):
        print('a')
        next_input, cells = add_lstm(input_dim,next_input)
        print(next_input)

    net = add_softmax_layer(next_input, input_dim, 2)
    return net

def getBidir(input_dim,input_var):
    """SimpleRecurrent-based bidirectionnal"""
    bidir = Bidirectional(weights_init=Orthogonal(),
                               prototype=SimpleRecurrent(
                                   dim=input_dim, activation=Tanh()))
    #bidir.allocate()
    bidir.initialize()
    h = bidir.apply(input_var)

    net = add_softmax_layer(h, input_dim, 2)

    return net

def getBidir2(input_dim,input_var):
    """ LSTM-based bidirectionnal """
    bidir = Bidirectional(weights_init=Orthogonal(),
                               prototype=LSTM(dim=input_dim, name='lstm'))
    #bidir.allocate()
    bidir.initialize()
    h = bidir.apply(input_var)

    net = add_softmax_layer(h, input_dim, 2)

    return net

def test():
    x = tensor.tensor3()
    b = getBidir(3,x)

    f = theano.function([x],b)

    print(f(np.ones((1,1,3),dtype=theano.config.floatX)))
    print(f(np.ones((1,1,3),dtype=theano.config.floatX)))
    print(f(np.ones((1,1,3),dtype=theano.config.floatX)))
    print(f(np.ones((1,1,3),dtype=theano.config.floatX)))



def test2():
    x = tensor.tensor3()
    b = getLSTMstack(3,x,5)

    f = theano.function([x],b)

    print(f(np.ones((1,1,3),dtype=theano.config.floatX)))
