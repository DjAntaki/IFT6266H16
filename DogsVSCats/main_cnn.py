#Initial code by Florian Bordes.
#Early-stopping, regularizations, configurations and some minors changes by Vincent Antaki
#Inspired by LeNet Mnist https://github.com/mila-udem/blocks-examples/blob/master/mnist_lenet/
import numpy as np
from theano import tensor as T
#Import functions from blocks
from blocks.roles import WEIGHT, BIAS
from blocks.filter import VariableFilter
from blocks.algorithms import GradientDescent, Scale, Adam, Momentum, RMSProp
from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence, Softmax)
from blocks.bricks import LeakyRectifier as Rectifier
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence, Flattener, MaxPooling, AveragePooling)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring, TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks_extras.extensions.plot import Plot
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import TrackTheBest

#Import functions from fuel
from toolz.itertoolz import interleave

from stream import get_stream

def get_config(config):
    config1={}

    if config == '5layers':
        config1['num_epochs'] = 150
        config1['num_channels'] = 3
        config1['image_shape'] = (192, 192)
        config1['filter_size'] = [(5,5),(5,5),(5,5),(5,5),(5,5)]
        config1['num_filter'] = [32, 48, 64, 128, 256]
        config1['pooling_sizes'] = [(2,2),(2,2),(2,2)]
        config1['mlp_hiddens'] = [1000,750,500]
        config1['output_size'] = 2
        config1['batch_size'] = 64
        config1['activation'] = [Rectifier() for _ in config1['num_filter']]
        config1['mlp_activation'] = [Rectifier().apply for _ in config1['mlp_hiddens']] + [Softmax().apply]
    elif config == '5layers':
        config1['num_epochs'] = 100
        config1['num_channels'] = 3
        config1['image_shape'] = (160, 160)
        config1['filter_size'] = [(5,5),(5,5),(5,5),(5,5),(5,5)]
        config1['num_filter'] = [32, 48, 64, 128, 256]
        config1['pooling_sizes'] = [(2,2),(2,2),(2,2)]
        config1['mlp_hiddens'] = [1000,750,500]
        config1['output_size'] = 2
        config1['batch_size'] = 64
        config1['activation'] = [Rectifier() for _ in config1['num_filter']]
        config1['mlp_activation'] = [Rectifier().apply for _ in config1['mlp_hiddens']] + [Softmax().apply]

    else :
        config1['num_epochs'] = 100
        config1['num_channels'] = 3
        config1['image_shape'] = (128, 128)
        config1['filter_size'] = [(5,5),(5,5),(5,5)]
        config1['num_filter'] = [20, 50, 80]
        config1['pooling_sizes'] = [(2,2),(2,2),(2,2)]
        config1['mlp_hiddens'] = [1000]
        config1['output_size'] = 2
        config1['batch_size'] = 64
        config1['activation'] = [Rectifier() for _ in config1['num_filter']]
        config1['mlp_activation'] = [Rectifier().apply for _ in config1['mlp_hiddens']] + [Softmax().apply]
        if config == 'test' :
            print("Test run...")
            config1['test'] = None
        else :
            print("Using default config..")

    return config1

def build_and_run(label, config):
    ############## CREATE THE NETWORK ###############
    #Define the parameters
    num_epochs, num_channels, image_shape, filter_size, num_filter, pooling_sizes, mlp_hiddens, output_size, batch_size, activation, mlp_activation  = config['num_epochs'], config['num_channels'], config['image_shape'], config['filter_size'], config['num_filter'], config['pooling_sizes'], config['mlp_hiddens'], config['output_size'], config['batch_size'], config['activation'], config['mlp_activation']
#    print(num_epochs, num_channels, image_shape, filter_size, num_filter, pooling_sizes, mlp_hiddens, output_size, batch_size, activation, mlp_activation)
    lambda_l1 = 0.0005
    lambda_l2 = 0.005

    print("Building model")
    #Create the symbolics variable
    x = T.tensor4('image_features')
    y = T.lmatrix('targets')

    #Get the parameters
    conv_parameters = zip(filter_size, num_filter)

    #Create the convolutions layers
    conv_layers = list(interleave([(Convolutional(
                                      filter_size=filter_size,
                                      num_filters=num_filter,
                                      name='conv_{}'.format(i))
                    for i, (filter_size, num_filter)
                    in enumerate(conv_parameters)),
                  (activation),
            (MaxPooling(size, name='pool_{}'.format(i)) for i, size in enumerate(pooling_sizes))]))
            #(AveragePooling(size, name='pool_{}'.format(i)) for i, size in enumerate(pooling_sizes))]))

    #Create the sequence
    conv_sequence = ConvolutionalSequence(conv_layers, num_channels, image_size=image_shape, weights_init=Uniform(width=0.2), biases_init=Constant(0.))
    #Initialize the convnet
    conv_sequence.initialize()
    #Add the MLP
    top_mlp_dims = [np.prod(conv_sequence.get_dim('output'))] + mlp_hiddens + [output_size]
    out = Flattener().apply(conv_sequence.apply(x))
    mlp = MLP(mlp_activation, top_mlp_dims, weights_init=Uniform(0, 0.2),
              biases_init=Constant(0.))
    #Initialisze the MLP
    mlp.initialize()
    #Get the output
    predict = mlp.apply(out)

    cost = CategoricalCrossEntropy().apply(y.flatten(), predict).copy(name='cost')
    error = MisclassificationRate().apply(y.flatten(), predict)

    #Little trick to plot the error rate in two different plots (We can't use two time the same data in the plot for a unknow reason)
    error_rate = error.copy(name='error_rate')
    error_rate2 = error.copy(name='error_rate2')

    ########### REGULARIZATION ##################
    cg = ComputationGraph([cost])
    weights = VariableFilter(roles=[WEIGHT])(cg.variables)
    biases = VariableFilter(roles=[BIAS])(cg.variables)
   # l2_penalty_weights = T.sum([i*lambda_l2/len(weights) * (W ** 2).sum() for i,W in enumerate(weights)]) # Gradually increase penalty for layer
    l2_penalty = T.sum([lambda_l2 * (W ** 2).sum() for i,W in enumerate(weights+biases)]) # Gradually increase penalty for layer
    #l2_penalty_bias = T.sum([lambda_l2*(B **2).sum() for B in biases])
    #l2_penalty = l2_penalty_weights + l2_penalty_bias
    l2_penalty.name = 'l2_penalty'
#    l1_penalty = T.sum([lambda_l1*T.abs_(z).sum() for z in weights+biases])
    l1_penalty_weights = T.sum([i*lambda_l1/len(weights) * T.abs_(W).sum() for i,W in enumerate(weights)]) # Gradually increase penalty for layer    
    l1_penalty_biases = T.sum([lambda_l1 * T.abs_(B).sum() for B in biases])
    l1_penalty = l1_penalty_biases + l1_penalty_weights
    l1_penalty.name = 'l1_penalty'
    costreg = cost + l2_penalty + l1_penalty
    costreg.name = 'costreg'
    
    ########### DEFINE THE ALGORITHM #############
#    algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Adam())
    algorithm = GradientDescent(cost=costreg, parameters=cg.parameters, step_rule=Momentum())

    ########### GET THE DATA #####################
    istest = 'test' in config.keys()
    train_stream, valid_stream, test_stream = get_stream(batch_size,image_shape,test=istest)
    

    ########### INITIALIZING EXTENSIONS ##########
    checkpoint = Checkpoint('models/best_'+label+'.tar')
    checkpoint.add_condition(['after_epoch'],
                         predicate=OnLogRecord('valid_error_rate_best_so_far'))
    #Adding a live plot with the bokeh server
    plot = Plot(label,
        channels=[['train_error_rate', 'valid_error_rate'],
                  ['valid_cost', 'valid_error_rate2'],
                  ['train_error_rate2','train_total_gradient_norm','train_l2_penalty','train_l1_penalty']], after_epoch=True,server_url="http://hades.calculquebec.ca:5042")

    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs),
                  DataStreamMonitoring([cost, error_rate, error_rate2], valid_stream, prefix="valid"),
                  TrainingDataMonitoring([costreg, error_rate, error_rate2,
                    aggregation.mean(algorithm.total_gradient_norm),l2_penalty,l1_penalty],
                    prefix="train",
                    after_epoch=True),
                  plot,
                  ProgressBar(),
                  Printing(),
                  TrackTheBest('valid_error_rate',min), #Keep best
                  checkpoint,  #Save best
                  FinishIfNoImprovementAfter('valid_error_rate_best_so_far', epochs=5)] # Early-stopping                  

    model = Model(cost)
    main_loop = MainLoop(algorithm,data_stream=train_stream,model=model,extensions=extensions)
    main_loop.run()


if __name__ == '__main__' :
    
    import sys
    print("Arguments :"+ ' '.join(sys.argv))
    
    if len(sys.argv) == 1 :
        config = get_config("default")
        label = 'default'
    elif '--test' in sys.argv:
        config = get_config("test")
        label = 'test'
    else :        
        config = get_config(sys.argv[1])
        label = '_'.join(sys.argv[1:])        
    build_and_run(label,config)
