import numpy as np

class ActiviationFunctions:

    def __init__(self, func):
        self.func = func
        
    ### ACTIVATION FUNCTIONS ###
    def activation(self, x):
        if self.func == 'relu':
            return np.maximum(0, x)
        elif self.func == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.func == 'softmax':
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation function: {self.func}")
    
    def activation_derivative(self, x):
        if self.func == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.func == 'sigmoid':
            return x * (1 - x)
        elif self.func == 'softmax':
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation function: {self.func}")
        