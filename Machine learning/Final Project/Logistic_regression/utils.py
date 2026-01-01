"""
Utility Functions for Logistic Regression

This module provides helper functions for data preprocessing,
evaluation metrics, and visualization.
"""

import numpy as np


def normalize_features(X, method='standardize'):
    """
    Normalize features for better gradient descent performance.
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, n_features)
        Input features
    method : str, default='standardize'
        Normalization method: 'standardize' or 'minmax'
        
    Returns:
    --------
    X_normalized : ndarray
        Normalized features
    params : dict
        Parameters used for normalization (mean, std, min, max)
    """
    X = np.array(X)
    
    if method == 'standardize':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        X_normalized = (X - mean) / std
        params = {'mean': mean, 'std': std}
        
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # Avoid division by zero
        X_normalized = (X - min_val) / range_val
        params = {'min': min_val, 'max': max_val}
        
    else:
        raise ValueError("Method must be 'standardize' or 'minmax'")
    
    return X_normalized, params


def apply_normalization(X, params, method='standardize'):
    """
    Apply previously computed normalization parameters to new data.
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, n_features)
        Input features
    params : dict
        Normalization parameters from normalize_features
    method : str, default='standardize'
        Normalization method: 'standardize' or 'minmax'
        
    Returns:
    --------
    ndarray
        Normalized features
    """
    X = np.array(X)
    
    if method == 'standardize':
        return (X - params['mean']) / params['std']
    elif method == 'minmax':
        return (X - params['min']) / (params['max'] - params['min'])
    else:
        raise ValueError("Method must be 'standardize' or 'minmax'")


def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
        
    Returns:
    --------
    dict
        Confusion matrix with keys: TP, TN, FP, FN
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Parameters:
    -----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
        
    Returns:
    --------
    dict
        Dictionary containing accuracy, precision, recall, f1_score
    """
    cm = confusion_matrix(y_true, y_pred)
    TP, TN, FP, FN = cm['TP'], cm['TN'], cm['FP'], cm['FN']
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': cm
    }


def print_metrics(metrics):
    """
    Print classification metrics in a readable format.
    
    Parameters:
    -----------
    metrics : dict
        Metrics dictionary from calculate_metrics
    """
    print("\n" + "="*50)
    print("CLASSIFICATION METRICS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  True Positives:  {cm['TP']}")
    print(f"  True Negatives:  {cm['TN']}")
    print(f"  False Positives: {cm['FP']}")
    print(f"  False Negatives: {cm['FN']}")
    print("="*50 + "\n")


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : ndarray
        Features
    y : ndarray
        Labels
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : ndarrays
        Split datasets
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.array(X)
    y = np.array(y)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
