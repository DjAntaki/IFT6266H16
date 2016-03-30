
#Model
from blocks.bricks import Tanh,Linear, Rectifier, Softmax, MLP, Logistic
from blocks.initialization import IsotropicGaussian, Constant, Uniform
from theano import tensor
from toolz.itertoolz import interleave
from itertools import product
    
from LeNet import LeNet
#from homemade_blocks import ResidualBrick

def mlp(input_size):
    """ an already done mlp for testing purposes"""
    return MLP(activations=[Rectifier(name='rect0'),Logistic(name='sigmoid_1'), 
        Softmax(name='softmax_2')], dims=[input_size, 1000, 500, 2],
        weights_init=IsotropicGaussian(), biases_init=Constant(0.01))
        
        
def convnet(input_size=None, feature_maps=None, mlp_hiddens=None,
         conv_sizes=None, pool_sizes=None):
    """LeNet from blocks-examples"""
    
    if feature_maps is None:
        feature_maps = [20, 50]
    if mlp_hiddens is None:
        mlp_hiddens = [500]
    if conv_sizes is None:
        conv_sizes = [5, 5]
    if pool_sizes is None:
        pool_sizes = [2, 2]
    if input_size is None :
        input_size = (150, 150)
    output_size = 2

    # Use ReLUs everywhere and softmax for the final prediction
    conv_activations = [Rectifier() for _ in feature_maps]
    mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()]
    convnet = LeNet(conv_activations, 3, input_size,
                    filter_sizes=zip(conv_sizes, conv_sizes),
                    feature_maps=feature_maps,
                    pooling_sizes=zip(pool_sizes, pool_sizes),
                    top_mlp_activations=mlp_activations,
                    top_mlp_dims=mlp_hiddens + [output_size],
                    border_mode='full',
                    weights_init=Uniform(width=.2),
                    biases_init=Constant(0))
    
    
    # We push initialization config to set different initialization schemes
    # for convolutional layers.
    convnet.push_initialization_config()
    convnet.layers[0].weights_init = Uniform(width=.2)
    convnet.layers[1].weights_init = Uniform(width=.09)
    convnet.top_mlp.linear_transformations[0].weights_init = Uniform(width=.08)
    convnet.top_mlp.linear_transformations[1].weights_init = Uniform(width=.11)
    convnet.initialize()

    return convnet


 
