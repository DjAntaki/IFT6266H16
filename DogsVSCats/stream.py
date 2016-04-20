def get_stream(batch_size, input_size, test=False):
    from fuel.datasets.dogs_vs_cats import DogsVsCats
    from fuel.streams import DataStream
    from fuel.schemes import ShuffledScheme
    from fuel.transformers.image import RandomFixedSizeCrop
    from fuel.transformers import Flatten #, ForceFloatX
    from ScikitResize import ScikitResize
    from fuel.transformers import Cast
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
