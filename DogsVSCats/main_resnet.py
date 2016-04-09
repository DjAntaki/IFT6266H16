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


def get_resnet_config(depth=1,image_size=(150,150),num_filters=32):
    assert depth>=0
    assert num_filters>0
    config1 = {}
    config1['label'] = str(depth)+'-bottledeep resnet'
    config1['depth'] = depth
    config1['num_filters'] = num_filters
    config1['image_size'] = image_size
    return config1

def get_config2(depth=2,image_size=(128,128),num_filters=32):
    assert depth>=0
    assert num_filters>0
    config1 = {}
    config1['label'] = str(depth)+'-bottledeep resnet'
    config1['depth'] = depth
    config1['num_filters'] = num_filters
    config1['image_size'] = image_size
    return config1

def get_test_resnet_config(label='test_resnet',depth=1,image_size=(128,128),num_filters=8):
    assert depth>=0
    assert num_filters>0
    config1 = {}
    config1['label'] = label
    config1['depth'] = depth
    config1['num_filters'] = num_filters
    config1['image_size'] = image_size
    return config1

def get_experiment_config(num_epochs=200, learning_rate=0.025,batch_size=32,num_batches=None,step_rule=None):
    config1 = {}
    assert num_epochs>0
    config1['num_epochs'] = num_epochs
    config1['batch_size'] = batch_size
    config1['num_batches'] = num_batches
    config1['step_rule'] = step_rule
    config1['learning_rate']= learning_rate
    config1['test'] = False
    return config1

def get_test_experiment_config(num_epochs=15, learning_rate=0.05,batch_size=5,num_batches=None, step_rule=None):

    config1 = {}
    assert num_epochs>0
    config1['num_epochs'] = num_epochs
    config1['batch_size'] = batch_size
    config1['num_batches'] = num_batches
    config1['step_rule'] = step_rule
    config1['learning_rate']= learning_rate
    config1['test'] = True
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
    l2_penality = lasagne.regularization.regularize_layer_params(layers[:len(layers)/3], lasagne.regularization.l1) * 1e-4
    loss = loss + l1_penality + l2_penality
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
    train_stream, valid_stream, test_stream = load_dataset1(experimentconfig['batch_size'],image_size,test=experimentconfig['test'])

    if 'step_rule' in experimentconfig.keys() and not experimentconfig['step_rule'] is None :
        step_rule = experimentconfig['step_rule'](learning_rate=experimentconfig['learning_rate'])
    else :
        step_rule=Scale(learning_rate=experimentconfig['learning_rate'])

    algorithm = GradientDescent(
                cost=loss, parameters=params,
                step_rule=step_rule)
       # step_rule=Scale(learning_rate=experimentconfig['learning_rate']))
      #  step_rule=Adam(learning_rate=experimentconfig['learning_rate']))
#        step_rule=RMSProp(learning_rate=experimentconfig['learning_rate']))
      #  step_rule=Momentum(learning_rate=experimentconfig['learning_rate']))

    grad_norm = aggregation.mean(algorithm.total_gradient_norm)    

    print("Initializing extensions...")
    checkpoint = Checkpoint('models/best_'+save_to+'.pkl')
    checkpoint.add_condition(['after_epoch'],
                         predicate=OnLogRecord('valid_acc_best_so_far'))

    #Defining extensions
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=experimentconfig['num_epochs'],
                              after_n_batches=experimentconfig['num_batches']),
                  TrainingDataMonitoring([loss, acc], prefix="train", after_n_epochs=1),
                  DataStreamMonitoring([loss, acc, grad_norm],valid_stream,prefix="valid", after_n_epochs=1),
                  #Checkpoint(save_to,after_n_epochs=5),
                  ProgressBar(),
                  #Plot(modelconfig['label'], channels=[['train_mean','test_mean'], ['train_acc','test_acc']], server_url='https://localhost:8007'), #'grad_norm'
                  #       after_batch=True),
                  Printing(every_n_epochs=1),
                  TrackTheBest('valid_acc'), #Keep best
                  checkpoint,  #Save best
                  FinishIfNoImprovementAfter('valid_acc_best_so_far', epochs=20)] # Early-stopping

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
    print(sys.argv)
    x = int(sys.argv[1])
    
    config = None 

    if('--test' in sys.argv):
        print("Retrieving test model config...")
        config = get_test_resnet_config(depth=1) 
        print("Retrieving test experiment config...")
        expr = get_test_experiment_config()
        print("Aweille Kevin continue comme ça...")
    else :
        if '--testmodel' in sys.argv:
            print("Retrieving test model config...") 
            config = get_test_resnet_config(depth=1,image_size=(50,50),num_filters=8)
        else :
            print("Retrieving default model config...")
            config = get_resnet_config2(depth=1,num_filters=8)
 
        if '--testexperiment' in sys.argv:    
            print("Retrieving test experiment config...") 
            expr = get_test_experiment_config()
        else :
            print("Retrieving default experiment config...")
            expr = get_experiment_config() 

    if x == 0 :
        build_and_run('testsave1.0-sgd',config,expr)
        pass    
    elif x == 1 :
        build_and_run('resenet1.1-sgd',config,expr)
    elif x == 2:
        from blocks.algorithms import Adam
        expr['step_rule'] = Adam
        build_and_run("resnet1.2-adam",config,expr)
    elif x == 3 :
        from blocks.algorithms import Momentum
        expr['step_rule'] = Momentum 
        build_and_run("resnet1.3-momentum",config,expr)
    elif x == 4 :
        from blocks.algorithms import RMSProp
        expr['step_rule'] = RMSProp
        build_and_run("resnet1.4-rmsprop2",config,expr)
 

