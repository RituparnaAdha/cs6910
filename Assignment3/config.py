config = {
  'train_path' : '/content/drive/MyDrive/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv',
  'val_path' : '/content/drive/MyDrive/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv',
  'test_path' : '/content/drive/MyDrive/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv',
  'hidden_layer_size':256,
  'num_encoder':2,
  'num_decoder':2,
  'embedding_size':16,
  'celltype':"rnn",
  'dropout':0.1,
  'beam_size':1,
  'batch_size':128,
  'epoch':1
}