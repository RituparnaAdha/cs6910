
## Problem Statement

The goal of this assignment is threefold: (i) train a CNN model from scratch and learn how to tune the hyperparameters and visualise filters (ii) finetune a pre-trained model just as you would do in many real world applications (iii) use an existing pre-trained model for a cool application.


## Prerequisites

```
Python 3.7.10
Numpy 1.19.5
keras 2.4.3
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

## Authors

 - [Shree Kanti](https://github.com/shreekanti/) 
 - [Rituparna Adha](https://github.com/RituparnaAdha/)

