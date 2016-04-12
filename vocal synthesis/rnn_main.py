
#Work in progress. Not functionnal yet.

import MFCC
import numpy as np
import theano

from theano import tensor
#from blocks.bricks.recurrent import BaseRecurrent, LSTM, Bidirectional, GatedRecurrent, SimpleRecurrent
#from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
#from blocks.bricks import Tanh
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.algorithms import GradientDescent, Scale
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing, Timing, ProgressBar
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.bricks.cost import SquaredError

def get_stream():
    from fuel.datasets.youtube_audio import YouTubeAudio
    data = YouTubeAudio('XqaJ2Ol5cC4')
    stream = data.get_example_stream()

    return stream

def get_test_expr_config():
    config = {}
    config['source_size'] = 4000
    config['target_size'] = 1000
    config['learning_rate']= 0.05
 #   config['batch_size'] = 2
    config['num_epochs'] = 10
    config['num_batches'] = None
    config['model'] = "LSTM4"
    return config

def get_expr_config(num_epochs=300,source_size=4000, target_size=1000, learning_rate=0.05,batch_size=64,num_batches=None):
    config = {}
    config['source_size'] = window_size
    config['target_size'] = target_size
    config['learning_rate']= learning_rate
#    config['batch_size'] = batch_size
    config['num_epochs'] = num_epochs
    config['num_batches'] = num_batches
    config['model'] = "LSTM4"
    return config

def main_rnn(config):

    x = tensor.matrix('features')
    y = tensor.vector('targets')

    if config['model'][:-1] == 'LSTM':
        from models import getLSTMstack
        y_hat = getLSTMstack(input_dim=2, input_var=x, depth=int(config['model'][-1]))
    else :
        raise Exception("lolwut")

#    y_hat = model.apply(x)

    #Cost
    cost = SquaredError().apply(y_hat ,y)
    #cost = CategoricalCrossEntropy().apply(T.flatten(),Y)
    cg = ComputationGraph(cost)

    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=Scale(learning_rate=config['learning_rate']))



    #Getting the stream
    train_stream = MFCC.get_stream(config['source_size'],config['target_size'])


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
 #       model=model,
        extensions=extensions)

    main_loop.run()

if __name__ == '__main__':
    main_rnn(get_test_expr_config())
