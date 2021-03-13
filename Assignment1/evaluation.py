import pickle
import assign1_q3 as nn

no_classes = 10
w_path = 'Assignment1/model/model-weights.pickle'
b_path = 'Assignment1/model/model-bias.pickle'

def load_wb(w_path,b_path):
    with open(w_path, 'rb') as f:
        weights = pickle.load(f)
    with open(b_path, 'rb') as f:
        bias = pickle.load(f)
    return weights, bias

if __name__ == '__main__':
    obj = nn.NN()
    obj.weights, obj.bias = load_wb(w_path,b_path)
    test_prediction = obj.make_predictions(nn.test_input_neurons)
    test_accuracy = obj.get_accuracy(test_prediction, nn.test_labels)
    print("test accuracy: {}".format(test_accuracy))


