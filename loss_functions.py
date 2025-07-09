import numpy as np

class LossFunctions:
    """
    A collection of loss functions for neural networks.
    """
    
    @staticmethod
    def cross_entropy(y_pred, y_true):
        """
        Calculate cross-entropy loss between predicted and expected vectors.
        Handles both single examples and batches of examples.
        
        Args:
            y_pred: Predicted probabilities from the network (output of softmax)
                    Shape can be (n_classes,) for a single example or (batch_size, n_classes) for a batch
            y_true: True labels (one-hot encoded)
                    Shape can be (n_classes,) for a single example or (batch_size, n_classes) for a batch
            
        Returns:
            loss: Cross-entropy loss value (scalar, averaged over the batch if batch input provided)
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Handle both single examples and batches
        if len(y_pred.shape) == 1:
            # Single example case
            loss = -np.sum(y_true * np.log(y_pred))
        else:
            # Batch case - sum across classes (axis=1), then average across batch
            loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        return loss
    
    @staticmethod
    def cross_entropy_derivative(y_pred, y_true):
        """
        Calculate derivative of cross-entropy loss with respect to predictions.
        Useful for backpropagation. Handles both single examples and batches.
        
        Args:
            y_pred: Predicted probabilities from the network (output of softmax)
                    Shape can be (n_classes,) for a single example or (batch_size, n_classes) for a batch
            y_true: True labels (one-hot encoded)
                    Shape can be (n_classes,) for a single example or (batch_size, n_classes) for a batch
            
        Returns:
            derivative: Derivative of cross-entropy loss with respect to predictions
                       Same shape as y_pred
        """
        # Add small epsilon to avoid division by zero
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Handle both single examples and batches
        if len(y_pred.shape) == 1:
            # Single example case
            derivative = -y_true / y_pred
        else:
            # Batch case - calculate derivative for each example in the batch
            # For softmax + cross-entropy, the derivative simplifies to (y_pred - y_true)
            # when using softmax activation and cross-entropy loss together
            derivative = -y_true / y_pred
            # Normalize by batch size to keep gradients at a similar scale regardless of batch size
            derivative = derivative / y_pred.shape[0]
        
        return derivative
