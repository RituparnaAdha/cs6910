
## Problem Statement

The goal of this assignment is threefold: (i) train a CNN model from scratch and learn how to tune the hyperparameters and visualise filters (ii) finetune a pre-trained model just as you would do in many real world applications (iii) use an existing pre-trained model for a cool application.

The assignment can be found [here](https://wandb.ai/miteshk/assignments/reports/Assignment-1--VmlldzozNjk4NDE?accessToken=r7ndsh8lf4wlxyjln7phvvfb8ftvc0n4lyn4tiowdg06hhzpzfzki4jrm28wqh44).

## Prerequisites

```
Python 3.7.10
Numpy 1.19.5
```
## Dataset
We have used the iNaturalist Dataset ("https://storage.googleapis.com/wandb_datasets/nature_12K.zip").

## Installing

 - Clone/download  this repository
 - For running in google colab, install wandb using following command -
  ``` !pip install wandb ```
 - For running locally, install wandb using following command  
  ``` 
  pip install wandb
  pip install numpy
  pip install keras
  ```
# Part A

# Part B
## Question 1 & 2

Solution Approach:
- We have used Keras library.
- The hyperparameters can be tweaked according the user's choice using the configurations dictionary.
- Make appropriate changes to the test, train, model folder names

The code for question 1 can be found [here](https://github.com/RituparnaAdha/cs6910/commit/5b82d7029dc1f46e8f102057e937477a8ea26e90).
The wandb visualisation for question 1 can be found [here](https://wandb.ai/shreekanti/assignment_1?workspace=user-shreekanti).


## Question 2
Solution Approach:
- Initialize  feedforward neural network`ffnn` class.
- Initialize randomazed weight and biases as per number of input, hidden and output layers specification.
- Implement activation functions such as Sigmoid, Softmax....
- Implement feedforward neural network algorithm.
The code for question 2 can be found [here](https://github.com/RituparnaAdha/cs6910/commit/349f0e600abf3c370c902df77675dbb2577d06aa).

## Question 3
Solution Approach:
- Get the output of feedforward backpropagation from previous question.
- Initialize one hot function to encode the labels of images.
- Implement backpropagation function.
- Initialize predictions, accurracy, loss, functions.
- Initialize gradient discent functions.
- Implement training function to use above functions.

The code for question 3 can be found [here](https://github.com/RituparnaAdha/cs6910/commit/81c7790be2c779fb9376a0158f4adb45645c70ec).

## Question 4

Solution Approach:

 - Split the train data in the ratio of 9:1. 90% of the data is for training purpose and 10% of the data is for validation.
 - Set the sweep function of wandb by setting up different parameters in sweep_config.
 - we can see the ouput within our wandb project using the code below-
```
wandb.agent(sweep_id,train)
```

The code for question 4 can be found [here](https://github.com/RituparnaAdha/cs6910/commit/3540f3753067f1dda62448578739f25d638d33c7).
The wandb visualisation for question 4 can be found [here](https://wandb.ai/shreekanti/confusion_matrix1/reports/Question-4--Vmlldzo1MjY2ODc).


## Question 5

The wandb visualisation for question 5 can be found [here](https://wandb.ai/rituparna_adha/assignement1/reports/Shared-panel-21-03-13-11-03-82--Vmlldzo1MjY2NzA).



## Question 6

The wandb visualisation for question 6 can be found [here](https://wandb.ai/rituparna_adha/assignement1/reports/Shared-panel-21-03-13-11-03-73--Vmlldzo1MjY2NzU).

## Question 7
Solution Approach:
- Get the best model.
- Report the best accuracy.
- The best model configuration is-
        learning_rate: 0.001,
	epochs: 10,
	no_hidden_layer: 3,
	size_hidden_layers:128,
	optimizer: adam,
	batch_size:128,
	activation: tanh,
	weight_initializations: random,
	weight_decay: 0,
	loss_function:ce

- Implement a function to calculate confusion matrix.
- Plot and integrate wandb to keep track using wandb.
The best model can be found [here](https://github.com/RituparnaAdha/cs6910/tree/main/Assignment1/model).
The code for question 7 can be found [here](https://github.com/RituparnaAdha/cs6910/commit/46a7deb1820b546099d0d6fb43afa8eacb6cdb34).
The wandb visualisation for question 7 can be found [here](https://wandb.ai/shreekanti/confusion_matrix1?workspace=user-shreekanti).
## Question 8
Solution Approach:
- Implement a function `squared error loss`.
- Get outputs of both `squared error loss` and `cross entropy loss`.
- Integrate the outputs of `squared error loss` and `cross entropy loss` to see automatically generated plot on wandb.

The code for question 8 can be found [here](https://github.com/RituparnaAdha/cs6910/commit/dada70a3e3ff58c3eb49839d602be272318946e5).
The wandb visualisation for question 8 can be found [here](https://wandb.ai/shreekanti/assignement1-lossfunc1?workspace=user-shreekanti).
## Evaluation

The code for evaluating our test data can be found [here](https://github.com/RituparnaAdha/cs6910/blob/main/Assignment1/evaluation.py).

## Report

The report for this assignment can be found [here](https://wandb.ai/shreekanti/confusion_matrix1/reports/Assignment1--Vmlldzo1MjY1MjU).
## Authors

 - [Shree Kanti](https://github.com/shreekanti/) 
 - [Rituparna Adha](https://github.com/RituparnaAdha/)

