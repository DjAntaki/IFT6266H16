
from config import *

from lasagne.layers import get_output, get_all_params
from lasagne.objectives import squared_error, categorical_crossentropy
from lasagne.updates import nesterov_momentum

from blocks.algorithms import GradientDescent, Scale
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.bricks.cost import MisclassificationRate

from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.monitoring import aggregation
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import TrackTheBest
from blocks_extras.extensions.plot import Plot
from theano import tensor as T
import theano
import lasagne
import string
import numpy as np
import vgg16

from stream import get_stream

def build_and_run(experimentconfig, modelconfig, save_to=None): #modelconfig, 
    """ part of this is adapted from lasagne tutorial""" 
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('image_features')
    target_var = T.lmatrix('targets')
    target_vec = T.extra_ops.to_one_hot(target_var[:,0],2)

    # Create vgg model
    print("Building model...")

    image_size = modelconfig['image_size']
    network = vgg16.build_small_model()
    prediction = lasagne.utils.as_theano_expression(lasagne.layers.get_output(network["prob"],input_var))
#    test_prediction = lasagne.layers.get_output(network["prob"],input_var,deterministic=True)

    # Loss function -> The objective to minimize 
    print("Instanciation of loss function...")
 
 #  loss = lasagne.objectives.categorical_crossentropy(prediction, target_var.flatten())
    loss = lasagne.objectives.squared_error(prediction,target_vec)
 #   test_loss = lasagne.objectives.squared_error(test_prediction,target_vec)
    loss = loss.mean()

    layers = network.values()  
    #l1 and l2 regularization
    pondlayers = {x:0.01 for x in layers}
    l1_penality = lasagne.regularization.regularize_layer_params_weighted(pondlayers, lasagne.regularization.l2)
    l2_penality = lasagne.regularization.regularize_layer_params(layers[len(layers)/4:], lasagne.regularization.l1) * 1e-4
    reg_penalty = l1_penality + l2_penality
    reg_penalty.name = 'reg_penalty'
    loss = loss + reg_penalty
    loss.name = 'reg_loss'

    error_rate = MisclassificationRate().apply(target_var.flatten(), prediction).copy(
            name='error_rate')

    # Load the dataset
    print("Loading data...")
    if 'test' in experimentconfig.keys() and experimentconfig['test'] is True:
        train_stream, valid_stream, test_stream = get_stream(experimentconfig['batch_size'],image_size,test=True)
    else :
        train_stream, valid_stream, test_stream = get_stream(experimentconfig['batch_size'],image_size,test=False)

    # Defining step rule and algorithm
    if 'step_rule' in experimentconfig.keys() and not experimentconfig['step_rule'] is None :
        step_rule = experimentconfig['step_rule'](learning_rate=experimentconfig['learning_rate'])
    else :
        step_rule=Scale(learning_rate=experimentconfig['learning_rate'])

    params = map(lasagne.utils.as_theano_expression,lasagne.layers.get_all_params(network['prob'], trainable=True))

    algorithm = GradientDescent(
                cost=loss, gradients={var:T.grad(loss,var) for var in params},
                step_rule=step_rule)

    grad_norm = aggregation.mean(algorithm.total_gradient_norm) 
    grad_norm.name='grad_norm'   

    print("Initializing extensions...")
    plot = Plot(save_to, channels=[['train_loss','valid_loss'], ['train_grad_norm','train_reg_penalty'],['train_error_rate','valid_error_rate']], server_url='http://hades.calculquebec.ca:5042')    
    checkpoint = Checkpoint('models/best_'+save_to+'.tar')
  #  checkpoint.add_condition(['after_n_batches=25'],
    checkpoint.add_condition(['after_epoch'],
                         predicate=OnLogRecord('valid_error_rate_best_so_far'))

    #Defining extensions
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=experimentconfig['num_epochs'],
                              after_n_batches=experimentconfig['num_batches']),
                  TrainingDataMonitoring([loss, error_rate, grad_norm, reg_penalty], prefix="train", after_epoch=True), #after_n_epochs=1
                  DataStreamMonitoring([loss, error_rate],valid_stream,prefix="valid", after_epoch=True), #after_n_epochs=1
                  #Checkpoint(save_to,after_n_epochs=5),
                  #ProgressBar(),
                  plot,
                  #       after_batch=True),
                  Printing(after_epoch=True),
                  TrackTheBest('valid_error_rate',min), #Keep best
                  checkpoint,  #Save best
                  FinishIfNoImprovementAfter('valid_error_rate_best_so_far', epochs=5)] # Early-stopping

   # model = Model(ComputationGraph(network))

    main_loop = MainLoop(
        algorithm,
        train_stream,
      #  model=model,
        extensions=extensions)
    print("Starting main loop...")

    main_loop.run()

build_and_run(get_expr_config('default'), vgg16.get_model("small"),save_to='test_vgg')