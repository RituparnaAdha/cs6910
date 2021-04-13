import wandb
from question1_2 import train

if __name__ == '__main__':
    sweep_config = {
        'method': 'bayes',
        "name": "assignement2",
        'metric':{
            'goal': 'maximize',
            'name': 'val_accuracy'
        },
        'parameters': {
        'epochs': {
            'values': [15,20]
        },
        'learning_rate':{
            'values': [0.001,0.01,0.0001,0.002,0.0002,0.02]
        },
        'optimizer':{
            'values': ['momentum','sgd','rmsprop','nesterov','adam','nadam']
        },
        'activation':{
            'values': ['relu']
        },
        'model_name':{
            'values': ['resnet','xception','inceptionv3', 'inceptionresnetv2']
        },
        'no_layers_to_freeze':{
            'values': [0,3,6,10]
        },
        'number_dense_layers':{
            'values':[0,1,2,3]
        },
        'dropout':{
            'values':[0.1,0.2,0.3]
        },
        {
            'l2':[0,0.01,0.001,0.005,0.05]
        }

    }
    }
    sweep_id = wandb.sweep(sweep_config)
    wandb.finish()