


config = {
  'train_path' : '/content/drive/MyDrive/dl_cs6910/assignment3/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv',
  'val_path' : '/content/drive/MyDrive/dl_cs6910/assignment3/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv',
  'test_path' : '/content/drive/MyDrive/dl_cs6910/assignment3/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv',
  'hidden_layer_size':256,
  'num_encoder':1,
  'num_decoder':1,
  'embedding_size':16,
  'celltype':"gru",
  'dropout':0.2,
  'beam_size':1,
  'batch_size':256,
  'epoch':30
}


    self.epoch =30 #config.get('epoch')
    self.batch_size = 256
    self.num_encoder_tokens = len(train_inp)+1#vocab size 
    self.num_decoder_tokens = len(train_tar)#vocab size
    self.max_encoder_seq_length = max([len(txt) for txt in train_text])
    self.max_decoder_seq_length = max([len(txt) for txt in train_target_text])
    self.input_token_index={'u': 0, 's': 1, 'b': 2, 'p': 3, 'k': 4, 'o': 5, 'v': 6, 'x': 7, 'r': 8, 'a': 9, 'h': 10, 'd': 11, 'z': 12, 'f': 13, 'n': 14, 'l': 15, 'g': 16, 'c': 17, 'y': 18, 'm': 19, 'w': 20, 'j': 21, 'q': 22, 'i': 23, 't': 24, 'e': 25, ' ': 26}
    self.target_token_index={'ई': 0, 'ङ': 1, 'ा': 2, 'ओ': 3, 'ट': 4, 'ध': 5, 'म': 6, 'फ': 7, 'द': 8, 'आ': 9, 'ं': 10, 'घ': 11, 'े': 12, 'ः': 13, 'श': 14, 'ॅ': 15, 'ग': 16, 'झ': 17, 'स': 18, 'ऐ': 19, '्': 20, 'न': 21, 'च': 22, '़': 23, 'ह': 24, 'ब': 25, 'ख': 26, 'थ': 27, 'औ': 28, 'ऋ': 29, 'य': 30, 'ढ': 31, 'ठ': 32, 'ड': 33, 'ऊ': 34, 'व': 35, 'इ': 36, 'ॉ': 37, 'छ': 38, 'ै': 39, 'ष': 40, 'ए': 41, 'ण': 42, 'ऑ': 43, 'त': 44, 'ु': 45, 'ो': 46, '\n': 47, 'प': 48, '\t': 49, 'र': 50, 'ृ': 51, 'अ': 52, 'ञ': 53, 'ू': 54, 'ज': 55, 'उ': 56, 'ी': 57, 'ौ': 58, 'ि': 59, 'ॐ': 60, 'ल': 61, 'क': 62, 'ँ': 63, 'भ': 64}

    self.train = self.encoding(train_text, train_target_text)
    self.validation = self.encoding(val_text, val_target_text)
    self.embedding_size = 16 #config.get('embedding_size')
    self.celltype = "gru"#onfig.get('celltype')
    self.number_encoder_layer= 1 #config.get('num_encoder')
    self.number_decoder_layer= 1#config.get('num_decoder')
    self.hidden_layers = 256#config.get('hidden_layer_size')
    self.lr = 0.01
    self.beam_size= 1#config.get('beam_size')
    self.dr= .2#config.get('dropout')