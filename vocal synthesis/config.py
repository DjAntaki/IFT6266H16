# Configs for LSTM experiment

def get_expr_config(conf):
    config = {}
    if conf == 'test':
        config['source_size'] = 4000
        config['target_size'] = 1000
        config['learning_rate']= 0.05
        config['batch_size'] = 5
        config['num_examples'] = 1000
        #config['num_epochs'] = 10
        config['num_batches'] = 30
        config['lstm_hidden_size'] = 50
        config['model'] = "LSTM4"
        config['learning_rate'] = 0.05
    elif conf == 'default':
        config['source_size'] = 4000
        config['target_size'] = 1000
        config['learning_rate']= 0.05
        config['batch_size'] = 32
        config['num_examples'] = 10000
        config['lstm_hidden_size'] = 100
#        config['num_epochs'] = 10
        config['num_batches'] = 3500
        config['model'] = "LSTM4"        
        config['learning_rate'] = 0.05  
    return config 
