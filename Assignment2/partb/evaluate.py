import question1_2 as net


if __name__ == "__main__":
  cnn = net.pretrained_model()
  cnn.load_model()
  cnn.predict()