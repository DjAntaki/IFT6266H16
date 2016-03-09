#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from argparse import ArgumentParser

from theano import tensor
from ScikitResize import ScikitResize
from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import (Rectifier, Initializable,
                           Softmax)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate, SquaredError
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.monitoring import aggregation
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from toolz.itertoolz import interleave
from itertools import product

def getconfig(num_epochs=1,input_resize=(150,150), learning_rate=0.05,batch_size=64,num_batches=None):
    config = {}
    config['num_epochs'] = num_epochs
    config['input_reshape'] = input_reshape
    config['learning_rate']= learning_rate
    config['batch_size'] = batch_size
    config['num_batches'] = num_batches
    return config

def main(save_to, model, train, test, num_epochs, input_size = (150,150), learning_rate=0.01,
batch_size=50, num_batches=None, flatten_stream=False):
    """ 
    save_to : where to save trained model
    model : model given in input must be already initialised (works with convnet and mlp)
    
    input_size : the shape of the reshaped image in input (before flattening is applied if flatten_stream is True)
    
    """
    if flatten_stream :
        x = tensor.matrix('image_features')
    else :
        x = tensor.tensor4('image_features')
    y = tensor.lmatrix('targets')

    #Data augmentation
    #insert data augmentation here 
    
    #Generating stream
    train_stream = DataStream.default_stream(
        train,
        iteration_scheme=ShuffledScheme(train.num_examples, batch_size)
    )

    test_stream = DataStream.default_stream(
        test,
        iteration_scheme=ShuffledScheme(test.num_examples, batch_size)
    )
    
    
    #Reshaping procedure
    #Add a crop option in scikitresize so that the image is not deformed
    
    #Resize to desired square shape
    train_stream = ScikitResize(train_stream, input_size, which_sources=('image_features',))
    test_stream = ScikitResize(test_stream, input_size, which_sources=('image_features',))
    
    #Flattening the stream
    if flatten_stream is True:
        train_stream = Flatten(train_stream, which_sources=('image_features',))
        test_stream = Flatten(test_stream, which_sources=('image_features',))
    
    # Apply input to model
    probs = model.apply(x)
    
    #Defining cost and various indices to watch
    #print(probs)
    #cost = SquaredError().apply(y.flatten(),probs)

    cost = CategoricalCrossEntropy().apply(y.flatten(), probs).copy(name='cost')
    error_rate = MisclassificationRate().apply(y.flatten(), probs).copy(
            name='error_rate')

    #Building Computation Graph
    cg = ComputationGraph([cost, error_rate])

    # Train with simple SGD
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=Scale(learning_rate=learning_rate))
    
    #Defining extensions
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs,
                              after_n_batches=num_batches),
                  TrainingDataMonitoring([cost, error_rate,aggregation.mean(algorithm.total_gradient_norm)], prefix="train", every_n_batches=5),
                  DataStreamMonitoring([cost, error_rate],test_stream,prefix="test", every_n_batches=25),
                  Checkpoint(save_to),
                  ProgressBar(),
                  Printing(every_n_batches=5)]

    # `Timing` extension reports time for reading data, aggregating a batch
    # and monitoring;
    # `ProgressBar` displays a nice progress bar during training.


    model = Model(cost)

    main_loop = MainLoop(
        algorithm,
        train_stream,
        model=model,
        extensions=extensions)

    main_loop.run()


def main_res_net():
    build_cnn()



if __name__ == '__main__':

    # Let's load and process the dataset
    import numpy as np
    from fuel.datasets.dogs_vs_cats import DogsVsCats

    from fuel.streams import DataStream
    from fuel.schemes import ShuffledScheme
    from fuel.transformers.image import RandomFixedSizeCrop
    from fuel.transformers import Flatten

    # Load the training set
    train = DogsVsCats(('train',),subset=slice(0, 20)) #subset=slice(0, 20000)
    test = DogsVsCats(('test',),subset=slice(0,20))
    input_size = (150,150)

    from models import mlp,convnet
    

    #main(None,mlp(input_size[0]*input_size[1]*3), train, test, num_epochs=1, input_size=input_size, batch_size=5, num_batches=20, flatten_stream=True)
    main("test1.txt", convnet(input_size), train, test, num_epochs=1, input_size=input_size, batch_size=64, num_batches=100)
    
#        from deep_res import build_cnn
 #       model = build_cnn(x,3,64)

# 
# THEANO_FLAGS='cuda.root=/usr/lib/nvidia-cuda-toolkit/', THEANO_FLAGS=cuda.root=/usr/lib/nvidia-cuda-toolkit/,device=gpu,floatX=float32 python dogs_cats.py
# THEANO_FLAGS=device=gpu   

# THEANO_FLAGS=exception_verbosity=high,optimizer=None
