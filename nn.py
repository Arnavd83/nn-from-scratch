import numpy as np
from activiation_functions import ActiviationFunctions
from loss_functions import LossFunctions

# Neural Network Project gang gang gang gang gang

class NeuralNetwork:

    # Set up network architecture
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Network architecture parameters (1 hidden layer)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights randomly
        self.weights = {
            'hidden': np.random.randn(self.input_size, self.hidden_size),
            'output': np.random.randn(self.hidden_size, self.output_size)
        }
        
        # Initialize biases to zero
        self.biases = {
            'hidden': np.zeros(self.hidden_size),
            'output': np.zeros(self.output_size)
        }

        # Initialize a cache to save intermediate steps within the forward pass for backpropagation
        self.cache = { 
            'hidden': {
                'z': None,
                'a': None
            },
            'output': {
                'z': None,
                'a': None
            }
        }

    # Going to restart the following allowing for better batching (to speed up training)
    # Also will have to take into account caching to use for backpropagation
    
    # Forward pass: Given set weight values and a batch of inputs find a corresponding batch of outputs
    def forward_pass(self, X):
        # X represents a batch on n inputs - dimensions: batch_size by inpute_features (784)
        
        # Find hidden layer:
        # z = XW + b
        # a = f(z)
        z_hidden = X @ self.weights['hidden'] + self.biases['hidden'] # (batch_size by hidden_size)
        a_hidden = ActiviationFunctions('relu').activation(z_hidden) # (batch_size by hidden_size)
        
        # Store in cache
        self.cache['hidden']['z'] = z_hidden 
        self.cache['hidden']['a'] = a_hidden
        
        # Find output layer:
        # z = aW + b
        # a = f(z)
        z_output = a_hidden @ self.weights['output'] + self.biases['output'] # (batch_size by output_size)
        a_output = ActiviationFunctions('softmax').activation(z_output) # (batch_size by output_size)
        
        # Store in cache
        self.cache['output']['z'] = z_output
        self.cache['output']['a'] = a_output
        
        # Return predictions
        predictions = a_output
        return predictions
    
    # Given a batch of predicted outputs and true outputs calculate the error using a loss function
    def calculate_error(self, prediction, targets):
        # Use the vectorized cross_entropy function that handles batches directly
        loss = LossFunctions.cross_entropy(prediction, targets)
        return loss

    
    # The goal of backpropagation if to find the gradient of the loss with respect to the weights and biases
    def backpropagation(self, X, y):
        ''' The math:
        For the output layer weights and biases: 
        dL/dW_out = dL/dA_out * dA_out/dZ_out * dZ_out/dW_out
        dL/dB_out = dL/dA_out * dA_out/dZ_out * dZ_out/dB_out

        For the hidden layer wieghts and biases:
        dL/dW_hidden = dL/dA_hidden * dA_hidden/dZ_hidden * dZ_hidden/dW_hidden
        dL/dB_hidden = dL/dA_hidden * dA_hidden/dZ_hidden * dZ_hidden/dB_hidden
            dL/DA_hidden = dL/dZ_out * dZ_out/DA_hidden
            dZ_out/DA_hidden = W_out because Z_out = A_hidden * W_out
            dL/dZ_out will be solved for in the earlier in the backpropogation steps

        '''
        # Find dL/dW_out
        dL_dAout = LossFunctions.cross_entropy_derivative(self.cache['output']['a'], y) # (batch_size, output_size)
        dAout_dZout = ActiviationFunctions('softmax').activation_derivative(self.cache['output']['z']) # (batch_size, output_size)
        # Use element wise multiplication because each node must remail its own entity in dL/dZ_out
        dL_dZout = dL_dAout * dAout_dZout # (batch_size, output_size)
        dZout_dWout = self.cache['hidden']['a'].T # (hidden_size, batch_size)
        dL_dWout = dL_dZout @ dZout_dWout # (output_size, hidden_size) -> this makes sense because Wout has dimenstions (output_size, hidden_size) 
        dL_dBout = np.sum(dL_dZout, axis=0) # (output_size,)


        

        
        


        




# Example usage
if __name__ == '__main__':
    from data_loader import DataLoader
    
    # Load and preprocess data
    data_loader = DataLoader()
    train_images, train_labels, val_images, val_labels, test_images, test_labels = data_loader.load_mnist_data()
    
    # Create neural network with the appropriate architecture
    nn = NeuralNetwork(784, 256, 10)
    

