from datasets import load_from_disk
from nn import *
import numpy as np
from loss_fn import *
import torch.nn as nn_torch
import torch

if __name__ == "__main__":

    ds = load_from_disk('nums.hf')

    input_shape = 784 

    hidden_layers = 2

    hidden_units = 128

    output_shape = 10

    nn = NeuralNet(input_shape, hidden_layers, hidden_units, output_shape)

    for i in range(len(ds['train'])):

        x = np.array(ds['train'][i]['image']).flatten()

        y_true = np.zeros(output_shape)
        
        y_true[ds['train'][i]['label']] = 1

        #print(y_true)

        output = nn.forward(x)
        
        loss_fn = nn_torch.CrossEntropyLoss()

        loss_torch = loss_fn(torch.tensor(output), torch.tensor(y_true))

        loss_me = cross_entropy_loss(output, y_true)

        print(f"NN output: {output} torch loss: {loss_torch} my loss: {loss_me}")

