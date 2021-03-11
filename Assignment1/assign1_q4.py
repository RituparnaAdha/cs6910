import wandb
from assign1_q3 import train

if __name__ == '__main__':
    sweep_config = {
        'method': 'bayes',
        "name": "assignement1-Sweep2",
        'metric':{
            'goal': 'maximize',
            'name': 'val_accuracy'
        },
        'parameters': {
        'epochs': {
            'values': [5,10,15]
        },
        'no_hidden_layers':{
            'values': [3,4,5]
        },
        'size_hidden_layers':{
            'values': [64,128]
        },
        'learning_rate':{
            'values': [0.001,0.01,0.0001,0.05,0.02]
        },
        'optimizer':{
            'values': ['momentum','sgd','rmsprop','nesterov','adam','nadam']
        },
        'batch_size':{
            'values': [32,64,128]
        },
        'activation':{
            'values': ['sigmoid','tanh','Relu']
        },
        'weight_initializations':{
            'values': ['random','xavier']
        },
        'weight_decay':{
            'values': [0,0.0005,0.05]
        }

    }
    }
    sweep_id = wandb.sweep(sweep_config,project ='assignement1-dummy')
    wandb.agent(sweep_id, function=train)