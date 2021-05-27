config = {
  'train_path' : '/content/drive/MyDrive/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv',
  'val_path' : '/content/drive/MyDrive/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv',
  'test_path' : '/content/drive/MyDrive/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv',
  'hidden_layer_size':128,
  'num_encoder':2,
  'num_decoder':1,
  'embedding_size':16,
  'celltype':"gru",
  'dropout':0,
  'beam_size':6,
  'batch_size':64,
  'epoch':30
}