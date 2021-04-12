import os
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
import keras
import tensorflow as tf
import PIL
from keras.applications.resnet50 import ResNet50 
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import wandb
from wandb.keras import WandbCallback
from keras.callbacks import EarlyStopping
from keras.regularizers import l2



train_folder ='gdrive/MyDrive/inaturalist_12K/train'
test_folder ='gdrive/MyDrive/inaturalist_12K/val'
model_folder ='./model'
IMG_DIM = (256,256,3)
batch_size =32
steps_per_epoch =250
validation_steps =30


configuration = {
    "model_name" : 'inceptionresnetv2', #  replace value of model with either of the 4 - xception,inceptionv3, inceptionresnetv2, resnet
    "num_classes" : 10,
    "no_layers_to_freeze" :0,
    "epochs": 15,
    "learning_rate": 0.001,
    "optimizer": 'momentum',
    "number_dense_layers": 5,
    "activation" : 'relu',
    "dropout":0.1,
    "l2": 0

}

def define_classes():
    classes =dict()
    class_=[f for f in os.listdir(train_folder) if f.startswith('.')!=1]
    for index,i in enumerate(class_):
        if (i.startswith('.')!=1):
            classes[index+1]= i
    return classes

class pretrained_model():
  
  def __init__(self, model_name= configuration.get("model_name"),num_classes= configuration.get("num_classes"),epochs= configuration.get("epochs"), num_frozen_layer = configuration.get("no_layers_to_freeze"), learning_rate =configuration.get("learning_rate"), optimizer = configuration.get("optimizer"),activation = configuration.get("activation"), no_dense_layers = configuration.get("number_dense_layers"),dropoutp = configuration.get("dropout"),l2 =configuration.get("l2")):
    self.model_name = model_name
    self.num_classes = num_classes
    self.classes = define_classes()
    self.model = []
    self.epochs = epochs
    self.num_frozen_layer = num_frozen_layer
    self.optimizer = optimizer
    self.lr = learning_rate
    self.activation = activation
    self.no_dense_layers = no_dense_layers
    self.dropout = dropoutp
    self.l2 = l2

    datagen = ImageDataGenerator(validation_split=0.1,rescale = 1./255.)
    self.train = datagen.flow_from_directory(
        train_folder, 
        subset='training',
        batch_size=batch_size,
        target_size=(256,256)
    )

    self.val = datagen.flow_from_directory(
        train_folder,
        subset='validation',
        batch_size=batch_size,
        target_size=(256,256)
    )
    datagen = ImageDataGenerator(rescale = 1./255.)
    self.test_it = datagen.flow_from_directory(test_folder, batch_size=batch_size,target_size=(256,256))




  def model_(self):

    if(self.model_name =='resnet'):
        model = ResNet50(weights='imagenet',include_top=False,input_shape=IMG_DIM)
    elif (self.model_name =='xception'):
        model = Xception(weights='imagenet',include_top=False,input_shape=IMG_DIM)
    elif (self.model_name =='inceptionv3'):
        model = InceptionV3(weights='imagenet',include_top=False,input_shape=IMG_DIM)
    elif (self.model_name =='inceptionresnetv2'):
        model = InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(IMG_DIM)

    if self.num_frozen_layer == -1:
      model.trainable = True
    else:
      for index,layer in enumerate(reversed(model.layers)):
          if index <self.num_frozen_layer:
            layer.trainable = True
          else:
            layer.trainable = False

    

    self.model = Sequential()
    self.model.add(model)
    self.model.add(Flatten())
    for i in range(0,self.no_dense_layers):
      self.model.add(Dense(pow(2,(10-i)), activation=self.activation,input_dim= self.model.output_shape,bias_regularizer=l2(self.l2)))
      self.model.add(Dropout(self.dropout))
    self.model.add(Dense(self.num_classes, activation='softmax',bias_regularizer=l2(self.l2)))


    if(self.optimizer == 'rmsprop'):
      self.model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(self.lr),
              metrics=['accuracy'])
    elif(self.optimizer == 'adam'):
      self.model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(self.lr),
              metrics=['accuracy'])
    elif(self.optimizer == 'nadam'):
      self.model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Nadam(self.lr),
              metrics=['accuracy'])
    elif(self.optimizer == 'sgd'):
      self.model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(self.lr),
              metrics=['accuracy'])
    elif(self.optimizer == 'momentum'):
      self.model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(self.lr,momentum = 0.9),
              metrics=['accuracy'])
    elif(self.optimizer == 'nesterov'):
      self.model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(self.lr,momentum = 0.9, nesterov=True),
              metrics=['accuracy'])
    
    self.model.summary()


  def finetune(self):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    with tf.device('/device:GPU:0'):
      self.model.fit_generator(self.train,steps_per_epoch =steps_per_epoch,epochs=self.epochs,verbose=1,validation_data=self.val,validation_steps=validation_steps,callbacks=[es])#,callbacks=[WandbCallback()])

  def save_model(self):
    self.model.save(model_folder+'/cnn.h5')

  def predict(self):
    score=self.model.evaluate(self.test_it,verbose=0)
    print("Test Accuracy:  ",score[1])

def train():
  wandb.init(config=configuration, magic=True,reinit = True)
  wandb.run.name = 'mn-'+wandb.config.model_name+'-no_layers_to_freeze-'+str(wandb.config.no_layers_to_freeze)+'-epochs-'+str(wandb.config.epochs)+'-dense-layers-'+str(wandb.config.number_dense_layers)+'-op-'+str(wandb.config.optimizer)
  print(wandb.run.name)
  model_name = wandb.config.model_name #  replace value of model with either of the 4 - xception,inceptionv3, inceptionresnetv2, resnet
  num_classes = wandb.config.num_classes
  no_layers_to_freeze = wandb.config.no_layers_to_freeze
  epochs= wandb.config.epochs
  learning_rate = wandb.config.learning_rate
  optimizer = wandb.config.optimizer
  activation = wandb.config.activation
  no_dense_layers = wandb.config.number_dense_layers
  dropoutp=wandb.config.dropout
  l2 = wandb.config.l2


  cnn = pretrained_model(model_name, num_classes,epochs, no_layers_to_freeze,learning_rate, optimizer,activation, no_dense_layers,dropoutp,l2)
  cnn.model_()
  cnn.finetune()
  cnn.save_model()
  cnn.predict()

if __name__ == "__main__":
  train()
  wandb.finish()


    