#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 09:38:41 2021

@author: jyoti
"""
import numpy as np

np.random.seed(0)
 
X =  [[1,2,3,2.5],
          [2,5,-1,2],
          [-1.5,2.7,3.3,-0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        #to initialise weights now for the shape of weight, we need to ask programmer number of inputs (4)(shape(,x)) and number of neurons (shape,(x,))
        #0.10 to normalise the weights
        self.biases = np.zeros((1,n_neurons)) #shape of the biases
        #shape of weight is actually n_neurons, n_inputs but we have done opposite here ;;the reason is when we do the forward pass
        # we do not need to do the transpose; hence no need of transpose
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        pass
    
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) #truncating att the values less than 0