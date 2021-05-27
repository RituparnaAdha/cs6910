


## Prerequisites

```
Python 3.7.10
Numpy 1.19.5
Tensorflow 2.4.1
```
## Dataset
We have used the iNaturalist Dataset ("https://github.com/google-research-datasets/dakshina").

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
- The hyperparameters can be tweaked according the user's choice using the config.py file.
- Make appropriate changes to the test, train, model folder names
- To run the code follow:
```
cd Assignment3
python question1.py
```
The code for question 1 can be found [here](https://github.com/RituparnaAdha/cs6910/blob/main/Assignment3/question1.py).
## Question 2
Solution Approach:
- Implemented wandb sweep to find best parameters and analyse various aspects
- To run the sweep code follow:
```
cd Assignment2
cd partb
python question3.py

```
The code for question 2 can be found [here](https://github.com/RituparnaAdha/cs6910/blob/main/Assignment3/question2.py). The wandb visualisation for question 2 can be found [here](https://wandb.ai/assignment3/assignment3-part1/sweeps/zujji1u4?workspace=user-rituparna_adha).
## Question 4
Solution Approach:

The code for question 4 can be found [here](https://github.com/RituparnaAdha/cs6910/blob/main/Assignment3/Dl3_q4.ipynb).

The instructions to run are instructed before every cell in the notebook

 
## Report

The report for this assignment can be found  [here](https://wandb.ai/rituparna_adha/uncategorized/reports/Assignment-2--Vmlldzo2MDYyOTA).
## Authors

 - [Shree Kanti](https://github.com/shreekanti/) 
 - [Rituparna Adha](https://github.com/RituparnaAdha/)