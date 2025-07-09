import numpy as np
import pandas as pd
import os
from data_loader import load_mnist
from activiation_functions import ActiviationFunctions
from loss_functions import LossFunctions

# Neural Network Project gang gang gang gang gang

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None
        self.test_images = None
        self.test_labels = None

        # Network architecrure paramters (1 hidden layer )
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

    # Given the wieghts/biases and input X compute the output of the network
    def _forward_propagation(self, X):
        # Initialize the activation function
        act_func = ActiviationFunctions('relu')
        # Compute the output of the network
        pre_activation_hidden = np.dot(X, self.weights['hidden']) + self.biases['hidden']
        hidden_layer = act_func.activation(pre_activation_hidden)
        pre_activation_output = np.dot(hidden_layer, self.weights['output']) + self.biases['output']
        output_layer = ActiviationFunctions('softmax').activation(pre_activation_output)
        return output_layer

    # Find the cross entropy loss of the network prediction compared to the expected output
    # MIGHT WANT TO TEST ON MULTIPLE ERROR FUNCTIONS
    def error_calculation(self, y_pred, y_true):
        # Validate that y_pred is a valid probability distribution (sums to 1 for each sample)
        row_sums = np.sum(y_pred, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-5):
            raise ValueError(f"Predicted values do not form valid probability distributions. Row sums: {row_sums}")
        
        # Validate that y_true is one-hot encoded (each row has exactly one 1 and the rest 0s)
        row_sums_true = np.sum(y_true, axis=1)
        if not np.allclose(row_sums_true, 1.0, rtol=1e-5, atol=1e-5):
            raise ValueError("True labels are not in one-hot encoding format (each row should sum to 1)")
            
        if not np.all(np.logical_or(np.isclose(y_true, 0.0), np.isclose(y_true, 1.0))):
            raise ValueError("True labels are not in one-hot encoding format (values should be 0 or 1)")
        
        return LossFunctions.cross_entropy(y_pred, y_true)


# Example usage
if __name__ == '__main__':
    nn = NeuralNetwork(784, 256, 10)
    nn.load_mnist_data()
    

