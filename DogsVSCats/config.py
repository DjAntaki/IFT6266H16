def get_model_config(config):
    config1={}
    if config == 'test':
        config1['depth'] = 1
        config1['num_filters'] = 8
        config1['image_size'] = (42,42)
        config1['num_blockstack'] = 1
    elif config == 'ModelA' or config == 'default':
        config1['depth'] = 1
        config1['num_filters'] = 8
        config1['image_size'] = (64,64)
        config1['num_blockstack'] = 3
    elif config == 'ModelB':
        config1['depth'] = 2
        config1['num_filters'] = 8
        config1['image_size'] = (64,64)
        config1['num_blockstack'] = 4
    elif config == 'ModelC':
        config1['depth'] = 2
        config1['num_filters'] = 8
        config1['image_size'] = (128,128)
        config1['num_blockstack'] = 4
    elif config == 'ModelD':
        config1['depth'] = 3
        config1['num_filters'] = 8
        config1['image_size'] = (128,128)
        config1['num_blockstack'] = 4
    else :
        print("Invalid config name")
        return
    return config1


def get_expr_config(config):
    config1={}
    if config == 'test':
        config1['num_epochs'] = 15
        config1['batch_size'] = 5
        config1['num_batches'] = None
        config1['learning_rate']= 0.05
        config1['step_rule'] = None
    elif config == 'conf1':
        config1['num_epochs'] = 200
        config1['batch_size'] = 16
        config1['num_batches'] = None
        config1['learning_rate']= 0.025
    elif config == 'conf2':
        config1['num_epochs'] = 200
        config1['batch_size'] = 32
        config1['num_batches'] = None
        config1['learning_rate']= 0.05
    else :
        if not config == 'default':
            print("Invalid config name. Using default.")
        config1['num_epochs'] = 80
        config1['batch_size'] = 32
        config1['num_batches'] = None
        config1['learning_rate']= 0.05
        config1['step_rule'] = None
    return config1


def get_resnet_config(depth=1,image_size=(150,150),num_filters=32):
    assert depth>=0
    assert num_filters>0
    config1 = {}
    config1['depth'] = depth
    config1['num_filters'] = num_filters
    config1['image_size'] = image_size
    return config1

def get_experiment_config(num_epochs=200, learning_rate=0.025,batch_size=32,num_batches=None,step_rule=None):
    config1 = {}
    assert num_epochs>0
    config1['num_epochs'] = num_epochs
    config1['batch_size'] = batch_size
    config1['num_batches'] = num_batches
    config1['step_rule'] = step_rule
    config1['learning_rate']= learning_rate
    config1['test'] = False
    return config1