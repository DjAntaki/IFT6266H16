#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from blocks.bricks.sequence_generators import Readout, SequenceGenerator, TrivialEmitter, AbstractEmitter
from blocks.bricks.recurrent import RecurrentStack
from blocks.utils import shared_floatx_zeros, shared_floatx
from blocks.bricks.base import application, lazy

from blocks.model import Model
from blocks.bricks.recurrent import LSTM
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant, Uniform, Identity


class TestEmitter(AbstractEmitter):
    """ """

    @lazy(allocation=['readout_dim'])
    def __init__(self,readout_dim,**kwargs):
        super(TestEmitter,self).__init__(**kwargs)
        self.readout_dim = readout_dim
#        self.cost_func = SquaredError()
#        self.cost_func = CategoricalCrossEntropy()
 #       self.children = [self.cost_func]

        pass

    @application
    def cost(self, readouts, outputs):
        return ((readouts - outputs)**2).sum(axis=-1)

#        out = self.cost_func.apply(readouts,outputs)
#        return out


    @application
    def emit(self, readouts):
        """ applique quelque chose sur l'output du réseau"""
        print("emit-readouts", readouts)
        return readouts
        
    @application
    def initial_outputs(self,batch_size):
        return tensor.zeros((batch_size, self.readout_dim), dtype=theano.config.floatX)

    def get_dim(self, name):
        if name == 'outputs':
            return self.readout_dim
        return super(TrivialEmitter, self).get_dim(name)


def main_rnn(config):

    x = tensor.tensor3('features')
    y = tensor.matrix('targets')

#    if 'LSTM' in config['model'] :
#        from models import getLSTMstack
#        y_hat = getLSTMstack(input_dim=13, input_var=x, depth=int(config['model'][-1]))
#    else :
#        raise Exception("These are not the LSTM we are looking for")

#    y_hat = model.apply(x)
    

    emitter = TestEmitter()
#    emitter = TrivialEmitter(readout_dim=config['lstm_hidden_size'])

#    cost_func = SquaredError()

 #   @application
 #   def qwe(self, readouts, outputs=None):
 #       print(type(self), type(readouts))
 #       x = cost_func.apply(readouts,outputs)
 #       return x
    print(type(emitter.cost))
 #   emitter.cost = qwe
  #  print(type(qwe))

    steps = 2 
    n_samples= config['target_size']

    transition = [LSTM(config['lstm_hidden_size']) for _ in range(4)]
    transition = RecurrentStack(transition,
            name="transition", skip_connections=False)

    source_names = [name for name in transition.apply.states if 'states' in name]

    readout = Readout(emitter, readout_dim=config['lstm_hidden_size'], source_names=source_names,feedback_brick=None, merge=None, merge_prototype=None, post_merge=None, merged_dim=None)

    seqgen = SequenceGenerator(readout, transition, attention=None, add_contexts=False)
    seqgen.weights_init = IsotropicGaussian(0.01)
    seqgen.biases_init = Constant(0.)
    seqgen.push_initialization_config()

    seqgen.transition.biases_init = IsotropicGaussian(0.01,1)
    seqgen.transition.push_initialization_config()
    seqgen.initialize()

    states = seqgen.transition.apply.outputs
    print('states',states)
    states = {name: shared_floatx_zeros((n_samples, config['lstm_hidden_size']))
        for name in states}

    cost_matrix = seqgen.cost_matrix(x, **states)
    cost = cost_matrix.mean()
    cost.name = "nll"

    cg = ComputationGraph(cost)
    model = Model(cost)
    #Cost
#    cost = SquaredError().apply(y_hat ,y)
    #cost = CategoricalCrossEntropy().apply(T.flatten(),Y)
 #   

        #for sampling
    #cg = ComputationGraph(seqgen.generate(n_steps=steps,batch_size=n_samples, iterate=True))
  

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


theano.scan() with iterate=False
