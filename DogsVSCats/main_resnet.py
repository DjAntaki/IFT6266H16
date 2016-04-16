#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deep_res import build_cnn
from config import *

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
from blocks.extensions.predicates import OnLogRecord
from blocks.monitoring import aggregation
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import TrackTheBest
#from blocks_extras.extensions.plot import Plot
from fuel.transformers import Cast

def load_dataset1(batch_size, input_size, test=False):

    from fuel.datasets.dogs_vs_cats import DogsVsCats
    from fuel.streams import DataStream
    from fuel.schemes import ShuffledScheme
    from fuel.transformers.image import RandomFixedSizeCrop
    from fuel.transformers import Flatten #, ForceFloatX
    from ScikitResize import ScikitResize
    
    # Load the training set
    if test :
        train = DogsVsCats(('train',),subset=slice(0, 200)) 
        valid = DogsVsCats(('train',),subset=slice(19800, 20000)) 
        test = DogsVsCats(('test',),subset=slice(0,4))
    else :
        train = DogsVsCats(('train',),subset=slice(0,22000)) 
        valid = DogsVsCats(('train',),subset=slice(22000, 25000)) 
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

    #ForceFloatX, to spare you from possible bugs
    #train_stream = ForceFloatX(train_stream)
    #valid_stream = ForceFloatX(valid_stream)
    #test_stream = ForceFloatX(test_stream)

    #Cast instead of forcefloatX
    train_stream = Cast(train_stream, dtype='float32',which_sources=('image_features',))
    valid_stream = Cast(valid_stream, dtype='float32',which_sources=('image_features',))
    test_stream = Cast(test_stream, dtype='float32',which_sources=('image_features',))


    return train_stream, valid_stream, test_stream

def augment_data():
    pass


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

def build_and_run(save_to,modelconfig,experimentconfig):
    """ part of this is adapted from lasagne tutorial""" 

    n, num_filters, image_size = modelconfig['depth'],modelconfig['num_filters'], modelconfig['image_size']
    
    print("Amount of bottlenecks: %d" % n)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('image_features')
    #target_var = T.ivector('targets')
    target_var = T.lmatrix('targets')
    target_vec = T.extra_ops.to_one_hot(target_var,2)

    # Create residual net model
    print("Building model...")
    network = build_cnn(input_var, image_size, n, num_filters)
    get_info(network)
    prediction = lasagne.layers.get_output(network)

    # Loss function -> The objective to minimize 
    print("Instanciation of loss function...")
 
 #  loss = lasagne.objectives.categorical_crossentropy(prediction, target_var.flatten())
    loss = lasagne.objectives.squared_error(prediction,target_vec)
    loss = loss.mean()
#    loss.name = 'x-ent_error'
#    loss.name = 'sqr_error'
    layers = lasagne.layers.get_all_layers(network)

    #l1 and l2 regularization
    pondlayers = {x:0.01 for x in layers}
    l1_penality = lasagne.regularization.regularize_layer_params_weighted(pondlayers, lasagne.regularization.l2)
    l2_penality = lasagne.regularization.regularize_layer_params(layers[len(layers)/5:], lasagne.regularization.l1) * 1e-4
    reg_loss = l1_penality + l2_penality
    reg_loss.name = 'reg_penalty'
    loss = loss + reg_loss
    loss.name = 'reg_sqr_error'

    params = lasagne.layers.get_all_params(network, trainable=True)

    #Accuracy    
    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var.flatten()),dtype=theano.config.floatX)
 #   acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_vec),dtype=theano.config.floatX)
    
    acc.name = 'acc'
    
    #cg = ComputationGraph(loss,parameters=params)
    #print(cg.variables)
#    print(cg.params)


#    cost = CategoricalCrossEntropy().apply(target_var.flatten(), prediction).copy(name='cost')
#    error_rate = MisclassificationRate().apply(target_var.flatten(), prediction).copy(
#            name='error_rate')
#    cg = ComputationGraph([cost])


#    print("Instantiation of live-plotting extention with bokeh-server...")
#    plot = Plot(modelconfig['label'], channels=[['train_mean','test_mean'], ['train_acc','test_acc']], server_url='https://127.0.0.1:8007/')    
    
    # Load the dataset
    print("Loading data...")
    if 'test' in experimentconfig.keys() :
        train_stream, valid_stream, test_stream = load_dataset1(experimentconfig['batch_size'],image_size,test=True)
    else :
        train_stream, valid_stream, test_stream = load_dataset1(experimentconfig['batch_size'],image_size,test=True)

    # Defining step rule and algorithm
    if 'step_rule' in experimentconfig.keys() and not experimentconfig['step_rule'] is None :
        step_rule = experimentconfig['step_rule'](learning_rate=experimentconfig['learning_rate'])
    else :
        step_rule=Scale(learning_rate=experimentconfig['learning_rate'])

    algorithm = GradientDescent(
                cost=loss, parameters=params,
                step_rule=step_rule)

    grad_norm = aggregation.mean(algorithm.total_gradient_norm)    

    print("Initializing extensions...")
    checkpoint = Checkpoint('models/best_'+save_to+'.tar')
  #  checkpoint.add_condition(['after_n_batches=25'],

    checkpoint.add_condition(['after_epoch'],
                         predicate=OnLogRecord('valid_acc_best_so_far'))

    #Defining extensions
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=experimentconfig['num_epochs'],
                              after_n_batches=experimentconfig['num_batches']),
                  TrainingDataMonitoring([loss, acc, grad_norm, reg_loss], prefix="train", after_epoch=True), #after_n_epochs=1
                  DataStreamMonitoring([loss, acc],valid_stream,prefix="valid", after_epoch=True), #after_n_epochs=1
                  #Checkpoint(save_to,after_n_epochs=5),
                  #ProgressBar(),
                  #Plot(modelconfig['label'], channels=[['train_mean','test_mean'], ['train_acc','test_acc']], server_url='https://localhost:8007'), #'grad_norm'
                  #       after_batch=True),
                  Printing(after_epoch=True),
                  TrackTheBest('valid_acc'), #Keep best
                  checkpoint,  #Save best
                  FinishIfNoImprovementAfter('valid_acc_best_so_far', epochs=5)] # Early-stopping

   # model = Model(ComputationGraph(network))

    main_loop = MainLoop(
        algorithm,
        train_stream,
      #  model=model,
        extensions=extensions)
    print("Starting main loop...")

    main_loop.run()

if __name__=='__main__':
    import sys
    #Usage notice,
    # [modelconfig] [experiment_config] [None|adam|Momentum|RMSProp]
    # 
    print("Arguments :"+ ' '.join(sys.argv))
    
    if('--test' in sys.argv):
        print("Retrieving test model config...")
        model_config = get_model_config('test') 
        print("Retrieving test experiment config...")
        expr_config = get_expr_config('test')
        print("Aweille Kevin continue comme ça...")
        label = 'test'
    else :        
        print("Retrieving test model config...")
        model_config = get_model_config(sys.argv[1])
        print("Retrieving test experiment config...")
        expr_config = get_expr_config(sys.argv[2])
        step = sys.argv[3]
        label = sys.argv[1] + '_' + sys.argv[2] + '_'+sys.argv[3]


        if step == 'adam' :
            from blocks.algorithms import Adam
            expr_config['step_rule'] = Adam
        elif step == 'momentum' :
            from blocks.algorithms import Momentum
            expr_config['step_rule'] = Momentum 
        elif step == 'RMSprop' :
            from blocks.algorithms import RMSProp
            expr_config['step_rule'] = RMSProp
        elif step == 'None':
            expr_config['step_rule'] = None
        else :
            print("No step defined, using RMSProp.")
            from blocks.algorithms import RMSProp
            expr_config['step_rule'] = RMSProp


    build_and_run(label,model_config,expr_config)

 

