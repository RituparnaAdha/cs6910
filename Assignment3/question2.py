import wandb
from question1 import begin

if __name__ == '__main__':
    sweep_config = {
        'method': 'bayes',
        'metric':{
            'goal': 'maximize',
            'name': 'val_accuracy'
        },
        'parameters': {
        'epoch': {
            'values': [50,30]
        },
        'hidden_layer_size':{
            'values': [128,256]
        },
        'celltype':{
            'values': ['rnn','lstm','gru']
        },
        'embedding_size':{
            'values': [16,32,64]
        },
        'num_encoder':{
            'values': [1,2,3,4]
        },
        'num_decoder':{
            'values': [1,2,3,4]
        },
        'beam_size':{
            'values':[1,3,6]
        },
        'dropout':{
            'values':[0.2,0.3,0]
        },
        
    }
    }
    sweep_id = wandb.sweep(sweep_config,entity="assignment3",project='assignment3-part1')#"6lzq4pe8"
    wandb.agent(sweep_id, function=begin,entity="assignment3",project='assignment3-part1')
    wandb.finish()