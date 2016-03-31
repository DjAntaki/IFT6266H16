#!/usr/bin/env python
# -*- coding: utf-8 -*-


###
#Examples - Not all of them are working
###

import numpy as np
import itertools
import theano
from theano import tensor

from blocks import initialization
from blocks.initialization import (
    Constant, IsotropicGaussian, Orthogonal, Identity, Uniform)
from blocks.bricks import Identity, Linear, Tanh
from blocks.bricks.recurrent import LSTM, SimpleRecurrent, GatedRecurrent,Bidirectional
from blocks.bricks.parallel import Fork

def example():
    """ Simple reccurent example. Taken from : https://github.com/mdda/pycon.sg-2015_deep-learning/blob/master/ipynb/blocks-recurrent-docs.ipynb """
    x = tensor.tensor3('x')

    rnn = SimpleRecurrent(dim=3, activation=Identity(), weights_init=initialization.Identity())
    rnn.initialize()
    h = rnn.apply(x)

    f = theano.function([x], h)
    print(f(np.ones((3, 1, 3), dtype=theano.config.floatX))) 

    doubler = Linear(
                 input_dim=3, output_dim=3, weights_init=initialization.Identity(2),
                 biases_init=initialization.Constant(0))
    doubler.initialize()
    h_doubler = rnn.apply(doubler.apply(x))

    f = theano.function([x], h_doubler)
    print(f(np.ones((3, 1, 3), dtype=theano.config.floatX))) 

    #Initial State
    h0 = tensor.matrix('h0')
    h = rnn.apply(inputs=x, states=h0)

    f = theano.function([x, h0], h)
    print(f(np.ones((3, 1, 3), dtype=theano.config.floatX),
            np.ones((1, 3), dtype=theano.config.floatX))) 


#
# Questions
#

def example2():
    """GRU"""
    x = tensor.tensor3('x')
    dim = 3

    fork = Fork(input_dim=dim, output_dims=[dim, dim*2],name='fork',output_names=["linear","gates"], weights_init=initialization.Identity(),biases_init=Constant(0))
    gru = GatedRecurrent(dim=dim, weights_init=initialization.Identity(),biases_init=Constant(0))

    fork.initialize()
    gru.initialize()

    linear, gate_inputs = fork.apply(x)
    h = gru.apply(linear, gate_inputs)

    f = theano.function([x], h)
    print(f(np.ones((dim, 1, dim), dtype=theano.config.floatX))) 

    doubler = Linear(
                 input_dim=dim, output_dim=dim, weights_init=initialization.Identity(2),
                 biases_init=initialization.Constant(0))
    doubler.initialize()

    lin, gate = fork.apply(doubler.apply(x))
    h_doubler = gru.apply(lin,gate)

    f = theano.function([x], h_doubler)
    print(f(np.ones((dim, 1, dim), dtype=theano.config.floatX))) 



def example3():
    """GRU + SimpleReccurrent. nope"""
    pass

def example4():
    """LSTM -> Plante lors de l'initialisation du lstm."""

    x = tensor.tensor3('x')
    dim=3

#    gate_inputs = theano.function([x],x*4)
    gate_inputs = Linear(input_dim=dim,output_dim=dim*4, name="linear",weights_init=initialization.Identity(), biases_init=Constant(2))

    lstm = LSTM(dim=dim,activation=Tanh(), weights_init=IsotropicGaussian(), biases_init=Constant(0))
    
    gate_inputs.initialize()
    hg = gate_inputs.apply(x)
    

    #print(gate_inputs.parameters)
    #print(gate_inputs.parameters[1].get_value())
    
    lstm.initialize()
    h, cells = lstm.apply(hg)
    print(lstm.parameters)
    
    f = theano.function([x], h)
    print(f(np.ones((dim, 1, dim), dtype=theano.config.floatX)))
    print(f(np.ones((dim, 1, dim), dtype=theano.config.floatX)))
    print(f(4*np.ones((dim, 1, dim), dtype=theano.config.floatX)))
 
    print("Good Job!")


#    lstm_output = 

    #Initial State
    h0 = tensor.matrix('h0')
    c =  tensor.matrix('cells')
    h,c1 = lstm.apply(inputs=x, states=h0, cells=c) # lstm.apply(states=h0,cells=cells,inputs=gate_inputs)

    f = theano.function([x, h0, c], h)
    print("a")
    print(f(np.ones((3, 1, 3), dtype=theano.config.floatX),
            np.ones((1, 3), dtype=theano.config.floatX),
            np.ones((1, 3), dtype=theano.config.floatX))) 


def example5():
    """Bidir + simplereccurent. Adaptation from a unittest in blocks """
    
    bidir = Bidirectional(weights_init=Orthogonal(),
                               prototype=SimpleRecurrent(
                                   dim=3, activation=Tanh()))
    
    simple = SimpleRecurrent(dim=3, weights_init=Orthogonal(),
                                  activation=Tanh(), seed=1)
    
    bidir.allocate()
    simple.initialize()
    
    bidir.children[0].parameters[0].set_value(
        
        simple.parameters[0].get_value())
    
    bidir.children[1].parameters[0].set_value(        
        simple.parameters[0].get_value())
    
    #Initialize theano variables and functions
    x = tensor.tensor3('x')
    mask = tensor.matrix('mask')
 
    calc_bidir = theano.function([x, mask],
                                 [bidir.apply(x, mask=mask)])
    calc_simple = theano.function([x, mask],
                                  [simple.apply(x, mask=mask)])
 

    #Testing time
 
    x_val = 0.1 * np.asarray(
        list(itertools.permutations(range(4))),
        dtype=theano.config.floatX)
        
    x_val = (np.ones((24, 4, 3), dtype=theano.config.floatX) *
                  x_val[..., None])
                  
    mask_val = np.ones((24, 4), dtype=theano.config.floatX)
    mask_val[12:24, 3] = 0

    h_bidir = calc_bidir(x_val, mask_val)[0]
    h_simple = calc_simple(x_val, mask_val)[0]
    h_simple_rev = calc_simple(x_val[::-1], mask_val[::-1])[0]
    

    print(h_bidir)
    print(h_simple)
    print(h_simple_rev)


#Example 6. Taken from blocks rnn tutorial

from blocks.bricks.recurrent import BaseRecurrent, recurrent

class FeedbackRNN(BaseRecurrent):
    def __init__(self, dim, **kwargs):
        super(FeedbackRNN, self).__init__(**kwargs)
        self.dim = dim
        self.first_recurrent_layer = SimpleRecurrent(
            dim=self.dim, activation=Identity(), name='first_recurrent_layer',
            weights_init=initialization.Identity())
        self.second_recurrent_layer = SimpleRecurrent(
            dim=self.dim, activation=Identity(), name='second_recurrent_layer',
            weights_init=initialization.Identity())
        self.children = [self.first_recurrent_layer,
                         self.second_recurrent_layer]

    @recurrent(sequences=['inputs'], contexts=[],
               states=['first_states', 'second_states'],
               outputs=['first_states', 'second_states'])
    def apply(self, inputs, first_states=None, second_states=None):
        first_h = self.first_recurrent_layer.apply(
            inputs=inputs, states=first_states + second_states, iterate=False)
        second_h = self.second_recurrent_layer.apply(
            inputs=first_h, states=second_states, iterate=False)
        return first_h, second_h

    def get_dim(self, name):
        return (self.dim if name in ('inputs', 'first_states', 'second_states')
                else super(FeedbackRNN, self).get_dim(name))

def example6():
    """ http://blocks.readthedocs.org/en/latest/rnn.html """
    x = tensor.tensor3('x')

    feedback = FeedbackRNN(dim=3)
    feedback.initialize()
    first_h, second_h = feedback.apply(inputs=x)

    f = theano.function([x], [first_h, second_h])
    for states in f(np.ones((3, 1, 3), dtype=theano.config.floatX)):
        print(states) 


#
# Experimentations
#

#Adaptation du code plus haut pour un nombre arbitraire de RNN
class FeedbackRNNStack(BaseRecurrent):
    depth = 3

    def __init__(self, dim, depth, **kwargs):
        super(FeedbackRNNStack, self).__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.children = []
        FeedbackRNNStack.depth = depth

        for i in range(depth):
            self.children.append(SimpleRecurrent(
            dim=self.dim, activation=Identity(), name=str(i)+'th_recurrent_layer',
            weights_init=initialization.Identity()))

    @recurrent(sequences=['inputs'], contexts=[],
               states=[str(i)+'th_states' for i in range(depth)],
               outputs=[str(i)+'th_states' for i in range(depth)])
    def apply(self, inputs):
        
        hs = [self.children[0].apply(inputs=inputs, iterate=False)]

        for i in range(1,self.depth):

            hs.append(self.children[i].apply(
            inputs=hs[-1], iterate=False))
        
        return hs

    def get_dim(self, name):
        return (self.dim if name in ['inputs'] + [str(i)+'th_states' for i in range(self.depth)]
                else super(FeedbackRNNStack, self).get_dim(name))


def example7():
    """Not working """
    x = tensor.tensor3('x')

    s = FeedbackRNNStack(dim=3,depth=3)
    s.initialize()
    hiddens = s.apply(inputs=x)

    f = theano.function([x], hiddens)
    for states in f(np.ones((3, 1, 3), dtype=theano.config.floatX)):
        print(states) 
     

def test_square():
    from blocks.bricks.cost import SquaredError
    x = tensor.tensor3()
    y = tensor.tensor3()

    c = SquaredError()
    o = c.apply(x,y)
    f = theano.function([x,y],o)
    print(f(np.ones((3,3,3),dtype=theano.config.floatX),5*np.ones((3,3,3),dtype=theano.config.floatX)))


#def dropout(X, p=0.):
#    if p > 0:
#        retain_prob = 1 - p
#        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
#        X /= retain_prob
#    return X

