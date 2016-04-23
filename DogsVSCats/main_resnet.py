#!/usr/bin/env python
# -*- coding: utf-8 -*-

from theano import tensor
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
from blocks.bricks.cost import MisclassificationRate, CategoricalCrossEntropy

from blocks.graph import ComputationGraph

from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.monitoring import aggregation
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import TrackTheBest
from blocks_extras.extensions.plot import Plot
from stream import get_stream

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

    n, num_filters, image_size, num_blockstack = modelconfig['depth'], modelconfig['num_filters'], modelconfig['image_size'], modelconfig['num_blockstack']
    
    print("Amount of bottlenecks: %d" % n)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('image_features')
    #target_value = T.ivector('targets')
    target_var = T.lmatrix('targets')
    target_vec = T.extra_ops.to_one_hot(target_var[:,0],2)
    #target_var = T.matrix('targets')
    # Create residual net model
    print("Building model...")
    network = build_cnn(input_var, image_size, n, num_blockstack, num_filters)
    get_info(network)
    prediction = lasagne.utils.as_theano_expression(lasagne.layers.get_output(network))
    test_prediction = lasagne.utils.as_theano_expression(lasagne.layers.get_output(network,deterministic=True))

    # Loss function -> The objective to minimize 
    print("Instanciation of loss function...")
 
    #loss = CategoricalCrossEntropy().apply(target_var.flatten(), prediction)
    #test_loss = CategoricalCrossEntropy().apply(target_var.flatten(), test_prediction)
 #   loss = lasagne.objectives.categorical_crossentropy(prediction, target_var.flatten()).mean()
  #  test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var.flatten()).mean()
    loss = lasagne.objectives.squared_error(prediction,target_vec).mean()
    test_loss = lasagne.objectives.squared_error(test_prediction,target_vec).mean()
  #  loss = tensor.nnet.binary_crossentropy(prediction, target_var).mean()
  #  test_loss = tensor.nnet.binary_crossentropy(test_prediction, target_var).mean()
    test_loss.name = "loss"

#    loss.name = 'x-ent_error'
#    loss.name = 'sqr_error'
    layers = lasagne.layers.get_all_layers(network)

    #l1 and l2 regularization
    #pondlayers = {x:0.000025 for i,x in enumerate(layers)}
    #l1_penality = lasagne.regularization.regularize_layer_params_weighted(pondlayers, lasagne.regularization.l2)
    #l2_penality = lasagne.regularization.regularize_layer_params(layers[len(layers)/4:], lasagne.regularization.l1) * 25e-6
    #reg_penalty = l1_penality + l2_penality
    #reg_penalty.name = 'reg_penalty'
    #loss = loss + reg_penalty
    loss.name = 'reg_loss'
    error_rate = MisclassificationRate().apply(target_var.flatten(), test_prediction).copy(
            name='error_rate')

    
    # Load the dataset
    print("Loading data...")
    istest = 'test' in experimentconfig.keys()
    if istest:
        print("Using test stream")
    train_stream, valid_stream, test_stream = get_stream(experimentconfig['batch_size'],image_size,test=istest)

    # Defining step rule and algorithm
    if 'step_rule' in experimentconfig.keys() and not experimentconfig['step_rule'] is None :
        step_rule = experimentconfig['step_rule'](learning_rate=experimentconfig['learning_rate'])
    else :
        step_rule=Scale(learning_rate=experimentconfig['learning_rate'])

    params = map(lasagne.utils.as_theano_expression,lasagne.layers.get_all_params(network, trainable=True))
    print("Initializing algorithm")
    algorithm = GradientDescent(
                cost=loss, gradients={var:T.grad(loss,var) for var in params},#parameters=cg.parameters, #params
                step_rule=step_rule)

    #algorithm.add_updates(extra_updates)


    grad_norm = aggregation.mean(algorithm.total_gradient_norm)
    grad_norm.name = "grad_norm"

    print("Initializing extensions...")
    plot = Plot(save_to, channels=[['train_loss','valid_loss'], 
['train_grad_norm'],
#['train_grad_norm','train_reg_penalty'],
['train_error_rate','valid_error_rate']], server_url='http://hades.calculquebec.ca:5042')    

    checkpoint = Checkpoint('models/best_'+save_to+'.tar')
  #  checkpoint.add_condition(['after_n_batches=25'],

    checkpoint.add_condition(['after_epoch'],
                         predicate=OnLogRecord('valid_error_rate_best_so_far'))

    #Defining extensions
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=experimentconfig['num_epochs'],
                              after_n_batches=experimentconfig['num_batches']),
                  TrainingDataMonitoring([test_loss, error_rate, grad_norm], # reg_penalty],
                  prefix="train", after_epoch=True), #after_n_epochs=1
                  DataStreamMonitoring([test_loss, error_rate],valid_stream,prefix="valid", after_epoch=True), #after_n_epochs=1
                  plot,
                  #Checkpoint(save_to,after_n_epochs=5),
                  #ProgressBar(),
             #     Plot(save_to, channels=[['train_loss','valid_loss'], ['train_error_rate','valid_error_rate']], server_url='http://hades.calculquebec.ca:5042'), #'grad_norm'
                  #       after_batch=True),
                  Printing(after_epoch=True),
                  TrackTheBest('valid_error_rate',min), #Keep best
                  checkpoint,  #Save best
                  FinishIfNoImprovementAfter('valid_error_rate_best_so_far', epochs=5)] # Early-stopping

 #   model = Model(loss)
 #   print("Model",model)


    main_loop = MainLoop(
        algorithm,
        train_stream,
       # model=model,
        extensions=extensions)
    print("Starting main loop...")

    main_loop.run()

if __name__=='__main__':
    import sys
    #Usage notice,
    # [modelconfig] [experiment_config] [None|adam|Momentum|RMSProp] optionnal:[extra_label]
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
        print("Retrieving model config...")
        model_config = get_model_config(sys.argv[1])
        print("Retrieving experiment config...")
        expr_config = get_expr_config(sys.argv[2])
        step = sys.argv[3]
        label = sys.argv[1] + '_' + sys.argv[2] + '_'+sys.argv[3]
        if len(sys.argv) > 4:
            label += sys.argv[4]

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

    print(model_config,expr_config,label)
    build_and_run(label,model_config,expr_config)

 

