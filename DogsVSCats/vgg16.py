# 
# I modified it a little.

# ORIGINAL FROM : https://github.com/Lasagne/Recipes

# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: non-commercial use only

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax

def get_model(config):
    config1 = {}
    if config == 'vgg8' :
        config1['image_size'] = (192,192)
        config1['init_filter'] = 48
        config1['depth'] = 4
        config1['num_hidden_layer_mlp'] = 2
        config1['size_hidden_layer_mlp'] = 10
    elif config == 'vgg16' :
        config1['image_size'] = (224,224)
        config1['init_filter'] = 64
        config1['depth'] = 5
        config1['num_hidden_layer_mlp'] = 2
        config1['size_hidden_layer_mlp'] = 10
    else :
        print("Using default/test config")
        config1['image_size'] = (128,128)
        config1['init_filter'] = 32
        config1['depth'] = 5
        config1['num_hidden_layer_mlp'] = 2
        config1['size_hidden_layer_mlp'] = 10

    return config1


def build_model():
    net = {}
    net['input_layer'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(
        net['input_layer'], 64, 3, pad=1)#, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1)#, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1)#, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1 )#, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1 )#, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1)#, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1)#, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1)#, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1)#, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1)#, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1)#, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1)#, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1)#, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=10)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=10)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=2, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net

def build_small_model():
    net = {}
    net['input_layer'] = InputLayer((None,3,128,128))
    net['conv1_1'] = ConvLayer(
        net['input_layer'], 64, 3, pad=1)#, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1)#, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1)#, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1 )#, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1 )#, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1)#, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_2'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1)#, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_1'], 2)
    net['fc6'] = DenseLayer(net['pool4'], num_units=200)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=100)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=2, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)
    return net

def build_medium_model():
    net = {}
    net['input_layer'] = InputLayer((None,3,128,128))
    net['conv1_1'] = ConvLayer(
        net['input_layer'], 64, 3, pad=1)#, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1)#, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1)#, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1 )#, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1 )#, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1)#, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_2'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1)#, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1)#, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_2'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1)#, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1)#, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1)#, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=10)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=10)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=2, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)
    return net
