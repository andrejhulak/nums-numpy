import numpy as np
from random import uniform

def ReLU(x):

    x[x <= 0] = 0

    return x

def softmax(x):

    exponents = [np.exp(i) for i in x]

    sum_of_exponents = sum(exponents)

    probabilities = [i / sum_of_exponents for i in exponents]

    return np.array(probabilities)

def init_radom_weights(shape1, shape2):

    bound = np.sqrt(6 / (shape1 + shape2))

    return np.random.uniform(-bound, bound, (shape1, shape2))

class NeuralNet(object):

    def __init__(self, input_shape, hidden_layers, hidden_units, output_shape):

        self.input_weights = init_radom_weights(input_shape, hidden_units)

        self.hidden_layers_dict = {}

        for i in range(hidden_layers):
            self.hidden_layers_dict[f'hidden_layer_{i}'] = init_radom_weights(hidden_units, hidden_units)

        self.output_layer = init_radom_weights(hidden_units, output_shape)

        self.bias_vec = np.random.rand(hidden_units + 2)

    def forward(self, x):
        
        x = self.input_weights.T @ x + self.bias_vec[0]

        x = ReLU(x)

        for i in range(len(self.hidden_layers_dict)):
            x = self.hidden_layers_dict[f'hidden_layer_{i}'].T @ x + self.bias_vec[i + 1]
            x = ReLU(x)

        x = self.output_layer.T @ x + self.bias_vec[-1]

        return x
    
    def backward(self, loss):
        pass