
  
## Problem Statement

The goal of this assignment is threefold: (i) train a CNN model from scratch and learn how to tune the hyperparameters and visualise filters (ii) finetune a pre-trained model just as you would do in many real world applications (iii) use an existing pre-trained model for a cool application.


## Prerequisites

```
Python 3.7.10
Numpy 1.19.5
Tensorflow 2.4.1
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
  - For using GPU in google colab, change runtime type to GPU and install GPU using following command
 ``` 
!pip install tensorflow-gpu
!nvidia-smi 
```
# Part A
## Question 1
Solution Approach:
- Initialize all parameters such as input shape, number of filters in each steps, activation function etc to get a flexible model.
- Define a model function according to the question instruction
- Return the model.
- Make a call to model function to get its summary which also contain the total number of parameter.
The code for question 1 can be found [here](https://github.com/RituparnaAdha/cs6910/blob/main/Assignment2/parta/partaq1.py).
## Question 2
Solution Approach:
Define a CNN class which contain a train function to return CNN model
- Define a train function which perform the following instructions
	- Initialize wandb and configure it.
	- Prepare training and validation dataset
	- init a object to get our model from CNN class
	- compile our model using training and test dataset
	- fit our model to train and get the history.
	- plot the matrices
	- save the models
- configure parameters to make use of sweep functionality provided by wandb.ai
- call sweep function provived by wandb to get its sweep id
- call wandb agent to run our model on different  combinations of parameters.
The code for question 2 can be found [here](https://github.com/RituparnaAdha/cs6910/blob/main/Assignment2/parta/partaq2.py). The wandb visualisation for question 1 can be found [here](https://wandb.ai/shreekanti/test_assignment2_test_run?workspace=user-shreekanti).
## Question 4
Solution Approach:
- Prepare the test dataset
- Load the best model by analyzing the automated plots of sweep functionality
- Call evaluation function of model to evaluate the test and get the test loss and accuracy
- Report the accuracy
- Initialize a variable which store all available class
- As per given instruction get prediction of test images by calling prediction function of our model.
- Plot the matrices to see the predictions
- Create a new model(filter) using tensorflow.keras.models, input and output  of best model to get  filter
- Fetch the first CNN layer to get the number of filters
- Predict the image using new model to get feature maps of the image
- Plot the feaure map to visualize all the filters.
The code for question 4 can be found [here](https://github.com/RituparnaAdha/cs6910/blob/main/Assignment2/parta/partaq4.py).The wandb visualisation for question 4 can be found [here](https://wandb.ai/shreekanti/partA_Q4?workspace=user-shreekanti).

 ## Question 5
Solution Approach:
- Load the best model
- Prepare the dataset
- Initialize a new guided model using tensorflow.keras.models
- Define a custom gradient(Guided Relu) function which allows  fine-grained control over the gradients for backpropagating non-negative gradients to have a more efficient or numerically stable gradient.
- Assign guided ReLU to all the Conv layers where activation is ReLU
- Use Gradient Tape to record the preprocessed input image for the forward pass which will help calculate the gradients for the backward pass. Use the GradientTape object to capture the gradients on the last Conv layer.
- Get the gradient of all(one batch size) input images.
- Plot the gradients of images
- Get the output shape of CONV5 layer which will help to vizualize the Guided back propagation
- Make a single neuron model using tensorflow.keras.models and shape of CONV5 layer output. Get gradient of input images and apply the above method to get gradient until the gradient length become 10.
- Plot the gradient images along with 10 neurons to visualise guided backprogation.
The code for question 5 can be found [here](https://github.com/RituparnaAdha/cs6910/blob/main/Assignment2/parta/partaq5.py).The wandb visualisation for question 5 can be found [here](https://wandb.ai/shreekanti/assignment2_partA_Q5?workspace=user-shreekanti).
# Part B
## Question 1 & 2

Solution Approach:
- We have used Keras library.
- The hyperparameters can be tweaked according the user's choice using the configurations dictionary.
- Make appropriate changes to the test, train, model folder names

The code for question 1 can be found [here](https://github.com/RituparnaAdha/cs6910/blob/main/Assignment2/partb/question1_2.py).


## Question 2
Solution Approach:
- Implemented various startegies of finetuning the pretrained model.
The code for question 2 can be found [here](https://github.com/RituparnaAdha/cs6910/blob/main/Assignment2/partb/question1_2.py).
- To train the model with best hyperparameters follow:
```
cd Assignment2
cd partb
python question1_2.py

```

## Question 3
Solution Approach:
- Implemented wandb sweep to find best parameters and analyse various aspects
- To run the sweep code follow:
```
cd Assignment2
cd partb
python question3.py

```

The code for question 3 can be found [here](https://github.com/RituparnaAdha/cs6910/blob/main/Assignment2/partb/question3.py).

## Evaluate

 - Code to check the test accuracy of the model
```
cd Assignment2
cd partb
python evaluate.py

```
## Report

The report for this assignment can be found  [here](https://wandb.ai/rituparna_adha/uncategorized/reports/Assignment-2--Vmlldzo2MDYyOTA).
## Authors

 - [Shree Kanti](https://github.com/shreekanti/) 
 - [Rituparna Adha](https://github.com/RituparnaAdha/)