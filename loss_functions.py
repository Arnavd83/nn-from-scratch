import numpy as np

class LossFunctions:
    """
    A collection of loss functions for neural networks.
    """
    
    @staticmethod
    def cross_entropy(y_pred, y_true):
        """
        Calculate cross-entropy loss between predicted and expected vectors.
        
        Args:
            y_pred: Predicted probabilities from the network (output of softmax)
            y_true: True labels (one-hot encoded)
            
        Returns:
            loss: Cross-entropy loss value
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calculate cross-entropy loss
        # For each sample, we sum -y_true * log(y_pred) across all classes
        loss = -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
        
        return loss
    
    @staticmethod
    def cross_entropy_derivative(y_pred, y_true):
        """
        Calculate derivative of cross-entropy loss with respect to predictions.
        Useful for backpropagation.
        
        Args:
            y_pred: Predicted probabilities from the network (output of softmax)
            y_true: True labels (one-hot encoded)
            
        Returns:
            derivative: Derivative of cross-entropy loss
        """
        # Add small epsilon to avoid division by zero
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Derivative of cross-entropy loss
        derivative = -y_true / y_pred / y_pred.shape[0]
        
        return derivative
