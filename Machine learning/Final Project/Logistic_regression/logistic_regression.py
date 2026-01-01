"""
Logistic Regression Implementation from Scratch

This module provides a complete implementation of logistic regression
using gradient descent optimization.
"""

import numpy as np


class LogisticRegression:
    """
    Logistic Regression classifier using gradient descent.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of iterations for gradient descent
    regularization : float, default=0.0
        L2 regularization parameter (lambda)
    verbose : bool, default=False
        If True, print cost during training
    
    Attributes:
    -----------
    weights : ndarray of shape (n_features,)
        Coefficients of the features
    bias : float
        Intercept term
    costs : list
        History of cost values during training
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=0.0, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.costs = []
    
    def _sigmoid(self, z):
        """
        Compute the sigmoid function.
        
        Parameters:
        -----------
        z : ndarray
            Input values
            
        Returns:
        --------
        ndarray
            Sigmoid of input values
        """
        # Clip values to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, X, y, weights, bias):
        """
        Compute the binary cross-entropy cost function with L2 regularization.
        
        Parameters:
        -----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
        weights : ndarray of shape (n_features,)
            Feature weights
        bias : float
            Bias term
            
        Returns:
        --------
        float
            Cost value
        """
        from scipy.sparse import issparse
        
        m = X.shape[0]
        
        # Forward propagation
        if issparse(X):
            z = np.asarray(X.dot(weights)).ravel() + bias
        else:
            z = np.dot(X, weights) + bias
            
        y_pred = self._sigmoid(z)
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross-entropy loss
        cost = -1/m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
        # Add L2 regularization term (don't regularize bias)
        if self.regularization > 0:
            reg_term = (self.regularization / (2 * m)) * np.sum(weights ** 2)
            cost += reg_term
        
        return cost
    
    def fit(self, X, y):
        """
        Fit the logistic regression model using gradient descent.
        
        Parameters:
        -----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values (0 or 1)
            
        Returns:
        --------
        self : object
            Fitted estimator
        """
        # Handle sparse matrices from scipy
        from scipy.sparse import issparse
        
        # Convert y to numpy array
        y = np.asarray(y).ravel()
        
        # Get dimensions
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.costs = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward propagation
            if issparse(X):
                z = np.asarray(X.dot(self.weights)).ravel() + self.bias
            else:
                z = np.dot(X, self.weights) + self.bias
            
            y_pred = self._sigmoid(z)
            
            # Compute gradients
            diff = y_pred - y
            if issparse(X):
                dw = (1/n_samples) * np.asarray(X.T.dot(diff)).ravel()
            else:
                dw = (1/n_samples) * np.dot(X.T, diff)
            
            db = (1/n_samples) * np.sum(diff)
            
            # Add regularization gradient (don't regularize bias)
            if self.regularization > 0:
                dw += (self.regularization / n_samples) * self.weights
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute and store cost
            if i % 100 == 0:
                cost = self._compute_cost(X, y, self.weights, self.bias)
                self.costs.append(cost)
                
                if self.verbose:
                    print(f"Iteration {i}: Cost = {cost:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        ndarray of shape (n_samples,)
            Predicted probabilities for the positive class
        """
        from scipy.sparse import issparse
        
        if issparse(X):
            z = np.asarray(X.dot(self.weights)).ravel() + self.bias
        else:
            z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict
        threshold : float, default=0.5
            Classification threshold
            
        Returns:
        --------
        ndarray of shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples,)
            True labels
            
        Returns:
        --------
        float
            Accuracy score
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
