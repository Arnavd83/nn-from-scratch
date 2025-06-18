import numpy as np
import pandas as pd
import os
from data_loader import load_mnist
from activiation_functions import ActiviationFunctions

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None
        self.test_images = None
        self.test_labels = None

        # Network architecrure paramters (1 hidden layer)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights randomly from 
        self.weights = {
            'hidden': np.random.randn(self.input_size, self.hidden_size),
            'output': np.random.randn(self.hidden_size, self.output_size)
        }
        
        # Initialize biases to zero
        self.biases = {
            'hidden': np.zeros(self.hidden_size),
            'output': np.zeros(self.output_size)
        }
        
    def load_mnist_data(self):
        # Load the data using the data_loader module
        train_images, train_labels, test_images, test_labels = load_mnist()
        
        ### DATA PREPROCESSING ###
        
        # Store the test data in class variables
        self.test_images = test_images
        self.test_labels = test_labels
        
        # Split training set into training (80%) and validation (20%) sets
        np.random.seed(42)  # For reproducibility
        n_train = len(train_images)
        indices = np.random.permutation(n_train)
        train_size = int(0.8 * n_train)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        self.train_images = train_images[train_indices]
        self.train_labels = train_labels[train_indices]
        self.val_images = train_images[val_indices]
        self.val_labels = train_labels[val_indices]
        
        # Normalize the image data (0-255 to 0-1)
        self.train_images = self.train_images.astype('float32') / 255
        self.val_images = self.val_images.astype('float32') / 255
        self.test_images = self.test_images.astype('float32') / 255
        
        # Flatten the images from 28x28 to 784-dimensional vectors
        self.train_images = self.train_images.reshape(-1, 28*28)
        self.val_images = self.val_images.reshape(-1, 28*28)
        self.test_images = self.test_images.reshape(-1, 28*28)
        
        # Convert labels to one-hot encoding
        self.train_labels = self._one_hot_encode(self.train_labels)
        self.val_labels = self._one_hot_encode(self.val_labels)
        self.test_labels = self._one_hot_encode(self.test_labels)
        
        return self.train_images, self.train_labels, self.val_images, self.val_labels, self.test_images, self.test_labels
    
    def _one_hot_encode(self, labels):
        """Convert labels to one-hot encoding"""
        n_samples = len(labels)
        n_classes = 10  # MNIST has 10 classes (0-9)
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), labels] = 1
        return one_hot

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def _sigmoid_derivative(self, x):
        return x * (1 - x)
        
    def _relu(self, x):
        return np.maximum(0, x)
        
    def _relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
        
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_propagation(self, X):
        """
        Perform forward propagation through the network.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            output: Network output after forward pass
        """
        # Input to hidden layer
        self.hidden_layer_input = np.dot(X, self.weights['hidden']) + self.biases['hidden']
        self.hidden_layer_output = self._relu(self.hidden_layer_input)
        
        # Hidden to output layer
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights['output']) + self.biases['output']
        self.output = self._softmax(self.output_layer_input)
        
        return self.output


# Example usage
if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.load_mnist_data()
    

