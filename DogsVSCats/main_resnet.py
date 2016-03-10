#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deep_res import build_cnn

from lasagne.layers import get_output, get_all_params
from lasagne.objectives import squared_error, categorical_crossentropy
from lasagne.updates import nesterov_momentum

from theano import tensor as T
import theano
import lasagne
import string
import numpy as np

from blocks.algorithms import GradientDescent, Scale
from blocks.main_loop import MainLoop
from blocks.model import Model

from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.monitoring import aggregation
from blocks.extensions.saveload import Checkpoint
from blocks_extras.extensions.plot import Plot


def load_dataset1(batch_size, input_size=(150,150), test=False):

    from fuel.datasets.dogs_vs_cats import DogsVsCats
    from fuel.streams import DataStream
    from fuel.schemes import ShuffledScheme
    from fuel.transformers.image import RandomFixedSizeCrop
    from fuel.transformers import Flatten
    from ScikitResize import ScikitResize
    
    # Load the training set
    if test :
        train = DogsVsCats(('train',),subset=slice(0, 10)) 
        valid = DogsVsCats(('train',),subset=slice(19996, 20000)) 
        test = DogsVsCats(('test',),subset=slice(0,10))
    else :
        train = DogsVsCats(('train',)) 
        valid = DogsVsCats(('train',),subset=slice(0, 1)) 
        test = DogsVsCats(('test',))


    #Generating stream
    train_stream = DataStream.default_stream(
        train,
        iteration_scheme=ShuffledScheme(train.num_examples, batch_size)
    )

    valid_stream = DataStream.default_stream(
        valid,
        iteration_scheme=ShuffledScheme(valid.num_examples, batch_size)
    )


    test_stream = DataStream.default_stream(
        test,
        iteration_scheme=ShuffledScheme(test.num_examples, batch_size)
    )
    
    
    #Reshaping procedure
    #Apply crop and resize to desired square shape
    train_stream = ScikitResize(train_stream, input_size, which_sources=('image_features',))
    valid_stream = ScikitResize(valid_stream, input_size, which_sources=('image_features',))
    test_stream = ScikitResize(test_stream, input_size, which_sources=('image_features',))

    return train_stream, valid_stream, test_stream


def get_resnet_config(depth=16,image_size=(150,150),num_filters=32):
    assert depth>=0
    assert num_filters>0
    config1 = {}
    config1['label'] = str(depth)+'-deep resnet'
    config1['depth'] = depth
    config1['num_filters'] = num_filters
    config1['image_size'] = image_size
    return config1

def get_test_resnet_config(label='test_resnet',depth=1,image_size=(150,150),num_filters=8):
    assert depth>=0
    assert num_filters>0
    config1 = {}
    config1['label'] = label
    config1['depth'] = depth
    config1['num_filters'] = num_filters
    config1['image_size'] = image_size
    return config1

def get_experiment_config(num_epochs=150, learning_rate=0.05,batch_size=8,num_batches=None):
    config1 = {}
    assert num_epochs>0
    config1['num_epochs'] = num_epochs
    config1['batch_size'] = batch_size
    config1['num_batches'] = num_batches
    config1['step_rule'] = None
    config1['learning_rate']= learning_rate
    return config1

def get_test_experiment_config(num_epochs=3, learning_rate=0.05,batch_size=2,num_batches=None):
    config1 = {}
    assert num_epochs>0
    config1['num_epochs'] = num_epochs
    config1['batch_size'] = batch_size
    config1['num_batches'] = num_batches
    config1['step_rule'] = None
    config1['learning_rate']= learning_rate
    return config1

def get_info(network):
    """taken from aljaro's residual network main. See file deep_res.py"""
    all_layers = lasagne.layers.get_all_layers(network)
    num_params = lasagne.layers.count_params(network)
    num_conv = 0
    num_nonlin = 0
    num_input = 0
    num_batchnorm = 0
    num_elemsum = 0
    num_dense = 0
    num_unknown = 0
    print("  layer output shapes:")
    for layer in all_layers:
        name = string.ljust(layer.__class__.__name__, 32)
        print("    %s %s" %(name, lasagne.layers.get_output_shape(layer)))
        if "Conv2D" in name:
            num_conv += 1
        elif "NonlinearityLayer" in name:
            num_nonlin += 1
        elif "InputLayer" in name:
            num_input += 1
        elif "BatchNormLayer" in name:
            num_batchnorm += 1
        elif "ElemwiseSumLayer" in name:
            num_elemsum += 1
        elif "DenseLayer" in name:
            num_dense += 1
        else:
            num_unknown += 1
    print("  no. of InputLayers: %d" % num_input)
    print("  no. of Conv2DLayers: %d" % num_conv)
    print("  no. of BatchNormLayers: %d" % num_batchnorm)
    print("  no. of NonlinearityLayers: %d" % num_nonlin)
    print("  no. of DenseLayers: %d" % num_dense)
    print("  no. of ElemwiseSumLayers: %d" % num_elemsum)
    print("  no. of Unknown Layers: %d" % num_unknown)
    print("  total no. of layers: %d" % len(all_layers))
    print("  no. of parameters: %d" % num_params)

def main():
    return build_and_run("test_resnet1",get_test_resnet_config(),get_test_experiment_config())

def build_and_run(save_to,modelconfig,experimentconfig):
    

    n, num_filters, image_size = modelconfig['depth'],modelconfig['num_filters'], modelconfig['image_size']
    
    print("Amount of bottlenecks: %d" % n)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('image_features')
    #target_var = T.ivector('targets')
    target_var = T.lmatrix('targets')
    #target_vec = T.extra_ops.to_one_hot(target_var,2)

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var, image_size, n, num_filters)
    get_info(network)
    prediction = lasagne.layers.get_output(network)

    print("Instanciation of loss function...")
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var.flatten())
#    loss = lasagne.objectives.categorical_crossentropy(prediction, target_vec)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    #updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    #test_prediction = lasagne.layers.get_output(network, deterministic=True)
    #test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
    #                                                        target_var)
    #test_loss = test_loss.mean()
    #test_loss.name = 'test_loss'
    
    # As a bonus, also create an expression for the classification accuracy:
    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var.flatten()),dtype=theano.config.floatX)
 #   acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_vec),dtype=theano.config.floatX)
    
    acc.name = 'acc'
    
    #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
    #                  dtype=theano.config.floatX)
    #test_acc.name = 'test_acc'
#    return test_prediction, prediction, loss, params

    #cg = ComputationGraph(loss,parameters=params)
    #print(cg.variables)
#    print(cg.params)


#    cost = CategoricalCrossEntropy().apply(target_var.flatten(), prediction).copy(name='cost')
#    error_rate = MisclassificationRate().apply(target_var.flatten(), prediction).copy(
#            name='error_rate')
#    cg = ComputationGraph([cost])

    print("Instantiation of live-plotting extention with bokeh-server...")
    plot = Plot(modelconfig['label'], channels=[['train_mean','test_mean'], ['train_acc','test_acc']], server_url='https://127.0.0.1:8007/')    
    # Load the dataset
    print("Loading data...")
    train_stream, valid_stream, test_stream = load_dataset1(experimentconfig['batch_size'])


    algorithm = GradientDescent(
                cost=loss, parameters=params,

        step_rule=Scale(learning_rate=experimentconfig['learning_rate']))

    #grad_norm = aggregation.mean(algorithm.total_gradient_norm)
    

    print("Initializing extensions...")
    #Defining extensions
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=experimentconfig['num_epochs'],
                              after_n_batches=experimentconfig['num_batches']),
                  TrainingDataMonitoring([loss, acc], prefix="train", every_n_batches=1),
                  DataStreamMonitoring([loss, acc],test_stream,prefix="test", every_n_batches=2),
                  Checkpoint(save_to),
                  ProgressBar(),
                  plot,
                  #Plot(modelconfig['label'], channels=[['train_mean','test_mean'], ['train_acc','test_acc']], server_url='https://localhost:8007'), #'grad_norm'
                  #       after_batch=True),
                  Printing(every_n_batches=1)]

   # model = Model(ComputationGraph(network))

    main_loop = MainLoop(
        algorithm,
        train_stream,
      #  model=model,
        extensions=extensions)
    print("Starting main loop...")

    main_loop.run()

if __name__=='__main__':
    main()
