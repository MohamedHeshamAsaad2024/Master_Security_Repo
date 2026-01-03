"""
Fake News Detection - Training Script

This script trains a logistic regression model on the preprocessed ISOT fake news dataset.
It uses the feature_pipeline API to load TF-IDF features that have already been generated.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directories to path to import modules
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Data_preprocessing_and_cleanup"))

from logistic_regression import LogisticRegression
from utils import calculate_metrics, print_metrics
from features_pipeline import load_feature_matrices
import joblib


def load_preprocessed_data(features_dir):
    """
    Load the preprocessed TF-IDF features and labels using feature_pipeline API.
    
    Parameters:
    -----------
    features_dir : str or Path
        Path to the directory containing preprocessed features
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    print("Loading preprocessed data using feature_pipeline API...")
    
    # Use the feature_pipeline API to load scaled features for Logistic Regression
    # scaled=True is used for LR and SVM models (see feature_pipeline.md line 91-92)
    X_train, X_test, y_train, y_test = load_feature_matrices(str(features_dir), scaled=True)
    
    print(f"  Training set: {X_train.shape[0]:,} samples, {X_train.shape[1]:,} features")
    print(f"  Test set: {X_test.shape[0]:,} samples, {X_test.shape[1]:,} features")
    print(f"  Training labels - Fake: {np.sum(y_train == 0):,}, Real: {np.sum(y_train == 1):,}")
    print(f"  Test labels - Fake: {np.sum(y_test == 0):,}, Real: {np.sum(y_test == 1):,}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, learning_rate=0.1, n_iterations=1000, regularization=0.01):
    """
    Train the logistic regression model.
    
    Parameters:
    -----------
    X_train : sparse matrix
        Training features
    y_train : array
        Training labels
    learning_rate : float
        Learning rate for gradient descent
    n_iterations : int
        Number of training iterations
    regularization : float
        L2 regularization parameter
        
    Returns:
    --------
    model : LogisticRegression
        Trained model
    """
    print("\nTraining Logistic Regression Model...")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Regularization (L2): {regularization}")
    print()
    
    model = LogisticRegression(
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        regularization=regularization,
        verbose=True
    )
    
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the model on training and test sets.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained model
    X_train, y_train : training data
    X_test, y_test : test data
    """
    print("\n" + "="*70)
    print("TRAINING SET EVALUATION")
    print("="*70)
    
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
    print_metrics(train_metrics)
    
    print("="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
    print_metrics(test_metrics)
    
    return train_metrics, test_metrics


def save_model(model, output_path):
    """
    Save the trained model to disk.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained model
    output_path : str
        Path to save the model
    """
    joblib.dump(model, output_path)
    print(f"\nModel saved to: {output_path}")


def main():
    """
    Main training pipeline.
    """
    print("="*70)
    print("FAKE NEWS DETECTION - TRAINING PIPELINE")
    print("="*70)
    print()
    
    # Paths (relative to this script's location)
    script_dir = Path(__file__).parent
    features_dir = script_dir.parent / "Data_preprocessing_and_cleanup" / "Output" / "features_out"
    model_save_path = script_dir / "trained_fake_news_model.pkl"
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data(features_dir)
    
    # Train model
    model = train_model(
        X_train, y_train,
        learning_rate=0.1,
        n_iterations=1000,
        regularization=0.01
    )
    
    # Evaluate model
    train_metrics, test_metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Save model
    save_model(model, model_save_path)
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nFinal Test Accuracy:    {test_metrics['accuracy']*100:.2f}%")
    print(f"Final Test Precision:   {test_metrics['precision']*100:.2f}%")
    print(f"Final Test Recall:      {test_metrics['recall']*100:.2f}%")
    print(f"Final Test Sensitivity: {test_metrics['sensitivity']*100:.2f}%")
    print(f"Final Test Specificity: {test_metrics['specificity']*100:.2f}%")
    print(f"Final Test F1-Score:    {test_metrics['f1_score']*100:.2f}%")
    if 'auc_roc' in test_metrics:
        print(f"Final Test AUC-ROC:     {test_metrics['auc_roc']:.4f}")
    print(f"\nModel saved as: {model_save_path}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
