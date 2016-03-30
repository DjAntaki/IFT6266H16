
#Work in progress. Not functionnal yet.


import theano
from theano import tensor

from fuel.transformers.sequences import Window
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ConstantScheme
from MFCC import MFCC


import numpy as np
from blocks.bricks.recurrent import BaseRecurrent, LSTM, Bidirectional, GatedRecurrent, SimpleRecurrent
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.bricks import Tanh

from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph
from fuel.streams import DataStream
from blocks.algorithms import GradientDescent, Scale
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring

from blocks.bricks.cost import SquaredError

def get_stream():
    from fuel.datasets.youtube_audio import YouTubeAudio
    

    data = YouTubeAudio('XqaJ2Ol5cC4')
    stream = data.get_example_stream()

    return stream

def getconfig(num_epochs=1,window_size=10,learning_rate=0.05,batch_size=64,num_batches=None):
    config = {}
    config['num_epochs'] = num_epochs
    config['window_size'] = window_size
    config['learning_rate']= learning_rate
    config['batch_size'] = batch_size
    config['num_batches'] = num_batches
    return config



def main_rnn(model,config):

    x = tensor.tensor3('x')
    y = tensor.tensor3()

    y_hat = model.apply(x)

    #Cost
    #classification_error= MisclassificationRate().apply(T.argmax(),Y)
    cost = SquaredError().apply(y_hat ,y)
    #cost = CategoricalCrossEntropy().apply(T.flatten(),Y)
    cg = ComputationGraph(cost)

    # Train with simple SGD
    # TODO:Uses momentum_step
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=Scale(learning_rate=config['learning_rate']))


    #Data preparation


    #Generating stream
    train_stream = get_stream()

    #MFCC and NGrams transform
    train_stream = MFCC(train_stream)

    train_stream = Window(1, config['window_size'], config['window_size'], True, train_stream)
    for source, target in train_stream.get_epoch_iterator():
        print(source, target)
        print(source[0])
        #train(source, target)


        #train_stream = Windows

    #Monitoring stuff
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=config['num_epochs'],
                              after_n_batches=config['num_batches']),
                  #DataStreamMonitoring([cost, error_rate],test_stream,prefix="test"),
                  TrainingDataMonitoring([cost], prefix="train", every_n_batches=1),
                  #Checkpoint(save_to),
                  ProgressBar(),
                  Printing(every_n_batches=1)]
   

    main_loop = MainLoop(
        algorithm,
        train_stream,
        model=model,
        extensions=extensions)

    main_loop.run()

if __name__ == '__main__':

    window_size = 10

    from models import getBidir

    model = getBidir(window_size)

    main_rnn(model, getconfig())
