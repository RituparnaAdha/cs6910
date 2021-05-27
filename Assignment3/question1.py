import numpy as np
import tensorflow as tf
import wandb
import config
import random
from tensorflow import keras

import ipywidgets as widgets

import numpy as np
import tensorflow as tf
import wandb
import config
import random
from tensorflow import keras
from tensorflow.keras.utils import plot_model

configuration = config.config
model_folder ='./model'

train_text =[]
val_text =[]

train_target_text =[]
val_target_text = []

train_inp = set()
val_inp = set()

train_tar = set()
val_tar = set()



with open(configuration.get("train_path"),'r') as f:
  lines = f.read().split("\n")
  for line in lines:
    try:
       target_text,input_text, _ = line.split("\t")
    except:
      continue
    target_text = "\t" + target_text + "\n"

    train_text.append(input_text)
    train_target_text.append(target_text)

    for char in input_text:
        if char not in train_inp:
            train_inp.add(char)
    for char in target_text:
        if char not in train_tar:
            train_tar.add(char)


with open(configuration.get("val_path"),'r') as f:
  lines = f.read().split("\n")
  
  for line in lines:
    try:
      target_text, input_text, _ = line.split("\t")
    except:
      continue
    #input_text, target_text, _ = line.split("\t")
    target_text = "\t" + target_text + "\n"

    val_text.append(input_text)
    val_target_text.append(target_text)

    for char in input_text:
        if char not in val_inp:
            val_inp.add(char)
    for char in target_text:
        if char not in val_tar:
            val_tar.add(char)






class RNN():
  def __init__(self):
    


    self.epoch = 50#wandb.config.epoch
    self.batch_size = wandb.config.batch_size
    self.num_encoder_tokens = len(train_inp)+1
    self.num_decoder_tokens = len(train_tar)
    self.max_encoder_seq_length = max([len(txt) for txt in train_text])
    self.max_decoder_seq_length = max([len(txt) for txt in train_target_text])
    #self.input_token_index, self.target_token_index = self.map_characters()
    self.input_token_index = {'j': 0, 'q': 1, 'i': 2, 'b': 3, 'm': 4, 'a': 5, 'o': 6, 'z': 7, 'u': 8, 'r': 9, 'd': 10, 'n': 11, 'c': 12, 'k': 13, 'l': 14, 'x': 15, 'v': 16, 'g': 17, 'w': 18, 'y': 19, 'p': 20, 'f': 21, 's': 22, 'h': 23, 't': 24, 'e': 25, ' ': 26}
    self.target_token_index = {'ढ': 0, 'ब': 1, 'ष': 2, 'ू': 3, '़': 4, 'ऊ': 5, 'आ': 6, 'ग': 7, 'ख': 8, 'ृ': 9, 'अ': 10, 'ु': 11, 'त': 12, 'ह': 13, 'ऑ': 14, 'ऐ': 15, 'घ': 16, 'ि': 17, 'द': 18, 'च': 19, 'ट': 20, 'ॉ': 21, 'ं': 22, 'म': 23, 'ओ': 24, 'ङ': 25, 'ई': 26, 'उ': 27, 'ल': 28, 'इ': 29, 'झ': 30, 'ॐ': 31, 'छ': 32, 'श': 33, 'औ': 34, 'भ': 35, 'क': 36, 'ड': 37, 'ज': 38, 'ा': 39, 'ः': 40, 'न': 41, 'र': 42, 'य': 43, 'फ': 44, 'ौ': 45, 'ए': 46, 'ऋ': 47, 'ो': 48, 'ै': 49, 'े': 50, 'ध': 51, '्': 52, 'ञ': 53, 'थ': 54, 'ँ': 55, 'ण': 56, 'व': 57, 'प': 58, 'ॅ': 59, 'ठ': 60, '\t': 61, 'स': 62, '\n': 63, 'ी': 64}


    self.train = self.encoding(train_text, train_target_text)
    self.validation = self.encoding(val_text, val_target_text)
    self.embedding_size =wandb.config.embedding_size
    self.celltype=wandb.config.celltype
    self.number_encoder_layer=wandb.config.num_encoder
    self.number_decoder_layer=wandb.config.num_decoder
    self.hidden_layers =wandb.config.hidden_layer_size
    self.lr = 0.001
    #self.beam_search=wandb.config.epoch
    self.beam_size=wandb.config.beam_size
    self.dr=wandb.config.dropout

    self.reverse_token_index, self.reverse_target_index = self.reverse_character_maps()

    self.tab_index=self.target_token_index['\t']

    print(self.tab_index,self.target_token_index['\n'])

  def map_characters(self):
    input_token_index = dict([(char, i) for i, char in enumerate(train_inp)])
    target_token_index = dict([(char, i) for i, char in enumerate(train_tar)])
    input_token_index[" "] = len(input_token_index)
    print(input_token_index)
    print(target_token_index)
    return input_token_index, target_token_index

  def reverse_character_maps(self):
    reverse_token_index = dict([( i,char) for i, char in enumerate(train_inp)])
    reverse_target_index = dict([( i,char) for i, char in enumerate(train_tar)])

    return reverse_token_index, reverse_target_index


  def encoding(self,text, target_text):
    decoder_target_data = np.zeros((len(text), self.max_decoder_seq_length, self.num_decoder_tokens), dtype="float32")
    encoder_input_data = np.zeros((len(text), self.max_encoder_seq_length), dtype="float32")
    decoder_input_data = np.zeros((len(text), self.max_decoder_seq_length), dtype="float32")

    for i, (input, target) in enumerate(zip(text, target_text)):
      for t, char in enumerate(input):
          encoder_input_data[i, t] = self.input_token_index[char]
      encoder_input_data[i, t + 1 :] =  self.input_token_index[" "]
      for t, char in enumerate(target):
          decoder_input_data[i, t] = self.target_token_index[char]
          if t > 0:
              decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.0
      decoder_input_data[i, t + 1 :] =  self.target_token_index["\n"]
      decoder_target_data[i, t:, self.target_token_index["\n"]] = 1.0

    return decoder_target_data, encoder_input_data, decoder_input_data
    



  def loss(self,x1,x2,y_true,training):
    y_predict = self.model([x1,x2],training)
    return y_predict,self.cce(y_true,y_predict)

  def grad(self, input1,input2, targets,training):
    with tf.GradientTape() as tape:
      logits,loss_value = self.loss( input1,input2, targets, training)
    return logits,loss_value, tape.gradient(loss_value, self.model.trainable_variables)

  def optimize(self,loss_tensor):
    opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
    #opt = tf.keras.optimizers.RMSprop(learning_rate=self.lr)
    train_op = opt.minimize(loss_tensor)

    return train_op

  def fill_feed_dict(self,dataset):
    target_output_data= dataset[0]
    train_data = dataset[1]
    target_data = dataset[2]

    start = [random.randint(0,train_data.shape[0]-1) for i in range(self.batch_size) ]

    encoder_input_data = train_data[start,:]
    decoder_input_data = target_data[start,:]
    decoder_output_data = target_output_data[start,:,:]


    return encoder_input_data, decoder_input_data,decoder_output_data,start

  def greedy_search(self,predictions):
    return tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

  def get_accuracy(self,prediction,start,type_):

    #result_text = []
    count =0
    print(type_)
    for i in range(prediction.shape[0]):
      string=''
      for j in range(prediction.shape[1]):
        if ((prediction[i,j] ==self.target_token_index['\n']) or (prediction[i,j] ==self.tab_index)):
          break;
          
        else:
          string = string + self.reverse_target_index[prediction[i,j]]
          
      if(type_=="train"):
        print(string, train_target_text[start[i]][1:-1])
        if(string == train_target_text[start[i]][1:-1]):
          print(string, train_target_text[start[i]][1:-1])
          count +=1
      elif(type_=="val"):
        print(string, val_target_text[start[i]][1:-1])
        if(string == val_target_text[start[i]][1:-1]):
          count +=1
      

      



    return count/prediction.shape[0]


  def beam_search_function(self,predictions):

    result,indices = tf.nn.top_k(predictions, self.beam_size)
    log_result = tf.math.log(result)

    y_pred = np.zeros((log_result.shape[0],self.max_decoder_seq_length ))

    for i in range(result.shape[0]):
      for j in range(result.shape[1]):
        if(j==0):
          pred = np.max(log_result[i,j,:].numpy())
          y_pred[i,j] = int(indices.numpy()[i,j,np.argmax(log_result[i,j,:].numpy()) ])
        else:
          pred_ = pred + log_result[i,j,:].numpy()
          pred = np.max(pred_)
          y_pred[i,j] =  int(indices.numpy()[i,j, np.argmax(pred_)] )
    return y_pred


  def rnn_model(self, e_input, d_input):

    encoder = tf.keras.layers.SimpleRNN(self.hidden_layers,return_sequences=True, return_state=True,dropout = self.dr)

    output, state = encoder(e_input)

    for i in range(self.number_encoder_layer-1):

      encoder = tf.keras.layers.SimpleRNN(self.hidden_layers,return_sequences=True, return_state=True,dropout = self.dr)

      output, state = encoder(output)
   
    decoder = tf.keras.layers.SimpleRNN(self.hidden_layers,return_sequences=True, return_state=True)

    d_output, d_state = decoder(d_input,initial_state=[state])

    for i in range(self.number_decoder_layer-1):

      decoder = tf.keras.layers.SimpleRNN(self.hidden_layers,return_sequences=True, return_state=True)

      d_output, d_state = decoder(d_output,initial_state=[state])

    decoder_dense = tf.keras.layers.Dense(self.num_decoder_tokens, activation="softmax")

    decoder_outputs = decoder_dense(d_output)

    return decoder_outputs, state



  def gru_model(self,e_input, d_input):

    encoder = tf.keras.layers.GRU(self.hidden_layers,return_sequences=True, return_state=True,dropout = self.dr)

    output, state = encoder(e_input)

    for i in range(self.number_encoder_layer-1):

      encoder = tf.keras.layers.GRU(self.hidden_layers,return_sequences=True, return_state=True,dropout = self.dr)

      output, state = encoder(output)
   
    decoder = tf.keras.layers.GRU(self.hidden_layers,return_sequences=True, return_state=True)

    d_output, d_state = decoder(d_input,initial_state=[state])

    for i in range(self.number_decoder_layer-1):

      decoder = tf.keras.layers.GRU(self.hidden_layers,return_sequences=True, return_state=True)

      d_output, d_state = decoder(d_output,initial_state=[state])

    decoder_dense = tf.keras.layers.Dense(self.num_decoder_tokens, activation="softmax")

    decoder_outputs = decoder_dense(d_output)

    return decoder_outputs,state

  def lstm_model(self,e_input, d_input):

    encoder = tf.keras.layers.LSTM(self.hidden_layers,return_sequences=True, return_state=True,dropout = self.dr)

    output, state,c = encoder(e_input)

    for i in range(self.number_encoder_layer-1):

      encoder = tf.keras.layers.LSTM(self.hidden_layers,return_sequences=True, return_state=True,dropout = self.dr)

      output, state,c = encoder(output)
   
    decoder = tf.keras.layers.LSTM(self.hidden_layers,return_sequences=True, return_state=True)

    d_output, d_state, d_c = decoder(d_input,initial_state=[state,c])

    for i in range(self.number_decoder_layer-1):

      decoder = tf.keras.layers.LSTM(self.hidden_layers,return_sequences=True, return_state=True)

      d_output, d_state, d_c = decoder(d_output,initial_state=[state, c])

    decoder_dense = tf.keras.layers.Dense(self.num_decoder_tokens, activation="softmax")

    decoder_outputs = decoder_dense(d_output)

    return decoder_outputs, [state,c]


  def save_model(self):
    self.model.save("model2")



  def run_training(self):
    with tf.device('/device:GPU:0'):
      
      input = keras.Input(shape=(self.max_encoder_seq_length),batch_size=self.batch_size)

      d_input = keras.Input(shape=(self.max_decoder_seq_length),batch_size=self.batch_size)

      e_embedding_output =tf.keras.layers.Embedding(input_dim=self.num_encoder_tokens, output_dim=self.embedding_size,input_length=self.max_encoder_seq_length)(input)

      d_embedding_output =tf.keras.layers.Embedding(self.num_decoder_tokens, self.embedding_size, input_length=self.max_decoder_seq_length)(d_input)



      if(self.celltype =="rnn"):

        decoder_outputs,state =self.rnn_model(e_embedding_output,d_embedding_output)

      elif(self.celltype =="gru"): 

        decoder_outputs,state =self.gru_model(e_embedding_output,d_embedding_output)


      elif(self.celltype =="lstm"): 

        decoder_outputs,state =self.lstm_model(e_embedding_output,d_embedding_output)


      

      self.model = tf.keras.Model([input, d_input], decoder_outputs)
      print(self.model.layers)

      self.cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
      self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
      self.epoch_loss_avg = tf.keras.metrics.Mean()
      self.train_metric=train_acc_metric = keras.metrics.CategoricalAccuracy()
      self.val_metric=train_acc_metric = keras.metrics.CategoricalAccuracy()
      plot_model(self.model, to_file='model.png')



      for epoch in range(self.epoch):
        prediction_decoder_input = np.zeros(self.train[2].shape)

        prediction_decoder_input[:,0] = self.target_token_index["\t"] 

        val_prediction_decoder_input = np.zeros(self.validation[2].shape)

        val_prediction_decoder_input[:,0] = self.target_token_index["\t"]

        for step in range(int(self.train[1].shape[0]/self.batch_size)):

            encoder_input_data, decoder_input_data,decoder_output_data,start  = self.fill_feed_dict(self.train)
            logits, loss_value, grads = self.grad(encoder_input_data, decoder_input_data,decoder_output_data,True)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

            self.epoch_loss_avg.update_state(loss_value)

            self.train_metric.update_state(decoder_output_data, logits)
        
            
            #predictions = self.model([self.train[1][:int(self.train[1].shape[0]/2)],prediction_decoder_input[:int(self.train[1].shape[0]/2)]],training=True)

        if( self.beam_size == 1):
            y_pred = self.greedy_search(logits)
            accuracy = self.get_accuracy(y_pred.numpy(),start,"train")
        else:
            y_pred = self.beam_search_function(logits)
            accuracy = self.get_accuracy(y_pred,start,"train")


        for step in range(int(self.validation[1].shape[0]/self.batch_size)):

          encoder_input_data, decoder_input_data,decoder_output_data ,start = self.fill_feed_dict(self.validation)

          val_predictions = self.model([encoder_input_data,decoder_input_data],training=False)

          self.val_metric.update_state(decoder_output_data, val_predictions) 

          #val_predictions = self.model([self.validation[1],val_prediction_decoder_input],training=False)
        val_loss = self.cce(decoder_output_data,val_predictions)

        if(self.beam_size == 1):
            y_pred = self.greedy_search(val_predictions)
            val_accuracy = self.get_accuracy(y_pred.numpy(),start,"val")
        else:
            y_pred = self.beam_search_function(val_predictions)
            val_accuracy = self.get_accuracy(y_pred,start,"val")

        print("cross_entropy_accuracy",self.train_metric.result().numpy())

        print("cross_entropy_val_accuracy",self.val_metric.result().numpy())


        print("epoch:  ",epoch, "   accuracy :  ", accuracy, "    loss: ",self.epoch_loss_avg.result().numpy(),"   val_accuracy :  ", val_accuracy, "  val_loss: ",val_loss.numpy())

        wandb.log({
        "epoch": epoch,
        "loss": self.epoch_loss_avg.result().numpy(),
        "accuracy": accuracy,
        "categorical_accuracy":self.train_metric.result().numpy(),
        "val_loss": val_loss.numpy(),
        "val_accuracy": val_accuracy,
        "categorical_val_accuracy":self.val_metric.result().numpy()})




def begin():
  wandb.init(project ='assignment3-part1',config=configuration, magic=True,reinit = True)    
  wandb.run.name = wandb.run.name = 'hl-'+str(wandb.config.hidden_layer_size)+'-ne-'+str(wandb.config.num_encoder)+'-nd-'+str(wandb.config.num_encoder)+'-es-'+str(wandb.config.embedding_size)+'-ct-'+str(wandb.config.celltype)+'-d-'+str(wandb.config.dropout)+'-bs-'+str(wandb.config.batch_size)+'-e-'+str(wandb.config.epoch)

  
  rnn=RNN()
  rnn.run_training()
  rnn.save_model()

if __name__ == "__main__":
  begin()
    