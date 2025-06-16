import numpy as np
import pandas as pd
import os
from data_loader import load_mnist

class NeuralNetwork:
    def __init__(self):
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        
    def load_mnist_data(self):
        # Load the data using the data_loader module
        train_images, train_labels, test_images, test_labels = load_mnist()
        
        ### DATA PREPROCESSING ###
        
        ### Store the data in class variables
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        
        # Normalize the image data (0-255 to 0-1)
        self.train_images = self.train_images.astype('float32') / 255
        self.test_images = self.test_images.astype('float32') / 255
        
        # Flatten the images from 28x28 to 784-dimensional vectors
        self.train_images = self.train_images.reshape(-1, 28*28)
        self.test_images = self.test_images.reshape(-1, 28*28)
        
        # Convert labels to one-hot encoding
        self.train_labels = self._one_hot_encode(train_labels)
        self.test_labels = self._one_hot_encode(test_labels)
        
        return self.train_images, self.train_labels, self.test_images, self.test_labels
    
    def _one_hot_encode(self, labels):
        """Convert labels to one-hot encoding"""
        n_samples = len(labels)
        n_classes = 10  # MNIST has 10 classes (0-9)
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), labels] = 1
        return one_hot

    def test_data_preprocessing(self, index=0):
        # Load and preprocess the data
        self.load_mnist_data()
        
        # Print the shape of the data
        print("\n=== Data Preprocessing Test ===")
        print(f"Training data shape: {self.train_images.shape}")
        print(f"Training labels shape: {self.train_labels.shape}")
        
        # Print an example training sample and its label
        print("\nExample training sample:")
        print(f"Image data (first 10 pixels): {self.train_images[index, :28*28]}")
        print(f"\nCorresponding one-hot encoded label: {self.train_labels[index]}")
        
        # Find the actual digit from the one-hot encoding
        digit = np.argmax(self.train_labels[index])
        print(f"\nThis represents digit: {digit}")
        
        # Verify the normalization
        print(f"\nData normalization check (should be between 0 and 1):")
        print(f"Min pixel value: {np.min(self.train_images[index])}")
        print(f"Max pixel value: {np.max(self.train_images[index])}")

# Example usage
if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.test_data_preprocessing()

