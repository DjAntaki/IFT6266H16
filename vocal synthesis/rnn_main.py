
#Work in progress. Not functionnal yet.

import MFCC
import config
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
from blocks.bricks.cost import SquaredError, CategoricalCrossEntropy


def main_rnn(config):

    x = tensor.tensor3('features')
    y = tensor.matrix('targets')

    if 'LSTM' in config['model'] :
        from models import getLSTMstack
        y_hat = getLSTMstack(input_dim=34, input_var=x, depth=int(config['model'][-1]))
    else :
        raise Exception("These are not the LSTM we are looking for")

#    y_hat = model.apply(x)

    #Cost
    cost = SquaredError().apply(y_hat ,y)
    #cost = CategoricalCrossEntropy().apply(T.flatten(),Y)
    cg = ComputationGraph(cost)

    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=Scale(learning_rate=config['learning_rate']))



    #Getting the stream
    train_stream = MFCC.get_stream(config['batch_size'],config['source_size'],config['target_size'],config['num_examples'])


    #Monitoring stuff
    extensions = [Timing(),
                  FinishAfter(after_n_batches=config['num_batches']),
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

def load_model(path):
    """Load the previously saved model"""
    pass

def generate_music():
    pass


if __name__ == '__main__':

  #  import sys
 #   if len(sys.argv) > 1 :

#    else :
    expr_config = config.get_expr_config('test')

    main_rnn(expr_config)
