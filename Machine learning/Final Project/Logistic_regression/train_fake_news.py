"""
Fake News Detection - Training Script

This script trains a logistic regression model on the preprocessed ISOT fake news dataset.
It uses the feature_pipeline API to load TF-IDF features that have already been generated.
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add parent directories to path to import modules
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Data_preprocessing_and_cleanup"))

from logistic_regression import LogisticRegression
from utils import calculate_metrics, print_metrics
from features_pipeline import load_feature_matrices, load_artifacts
import joblib


def load_config(config_path='config.json'):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {config_path} not found. Using default hyperparameters.")
        return {
            'hyperparameters': {
                'learning_rate': 0.1,
                'n_iterations': 1000,
                'regularization': 0.01
            }
        }



def plot_confusion_matrix(metrics, dataset_name='Test', save_path=None):
    """
    Plot confusion matrix visualization.
    
    Parameters:
    -----------
    metrics : dict
        Metrics dictionary containing 'confusion_matrix'
    dataset_name : str
        Name of the dataset (for title)
    save_path : str or Path, optional
        Path to save the plot
    """
    cm = metrics['confusion_matrix']
    
    # Create confusion matrix array
    conf_matrix = np.array([
        [cm['TN'], cm['FP']],
        [cm['FN'], cm['TP']]
    ])
    
    # Create figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake (0)', 'Real (1)'],
                yticklabels=['Fake (0)', 'Real (1)'],
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {dataset_name} Set', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add accuracy text
    accuracy = metrics['accuracy']
    plt.text(0.5, -0.15, f"Accuracy: {accuracy*100:.2f}%", 
             ha='center', transform=plt.gca().transAxes, fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved confusion matrix to: {save_path}")
    
    plt.close()


def plot_learning_curve(model, save_path=None):
    """
    Plot learning curve showing cost over iterations.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained model with cost history
    save_path : str or Path, optional
        Path to save the plot
    """
    if not hasattr(model, 'costs') or len(model.costs) == 0:
        print("  Warning: No cost history available")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot cost curve
    iterations = [i * 100 for i in range(len(model.costs))]
    plt.plot(iterations, model.costs, 'b-', linewidth=2, label='Training Cost')
    
    plt.title('Learning Curve - Cost vs Iterations', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost (Binary Cross-Entropy)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add final cost text
    final_cost = model.costs[-1]
    plt.text(0.98, 0.95, f'Final Cost: {final_cost:.6f}', 
             ha='right', va='top', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved learning curve to: {save_path}")
    
    plt.close()


def plot_feature_importance(model, features_dir, top_n=20, save_path=None):
    """
    Plot feature importance based on model weights.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained model with weights
    features_dir : str or Path
        Directory containing artifacts (to get feature names)
    top_n : int
        Number of top features to show
    save_path : str or Path, optional
        Path to save the plot
    """
    # Load artifacts to get feature names
    artifacts = load_artifacts(str(features_dir))
    tfidf = artifacts['tfidf']
    feature_names = tfidf.get_feature_names_out()
    
    # Get model weights
    weights = model.weights
    
    # Get top positive and negative features
    top_positive_idx = np.argsort(weights)[-top_n:][::-1]
    top_negative_idx = np.argsort(weights)[:top_n]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot top positive features (predicting Real news)
    positive_features = [feature_names[i] for i in top_positive_idx]
    positive_weights = weights[top_positive_idx]
    
    ax1.barh(range(top_n), positive_weights, color='green', alpha=0.6)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(positive_features, fontsize=9)
    ax1.set_xlabel('Weight (Coefficient)', fontsize=11)
    ax1.set_title(f'Top {top_n} Features Predicting REAL News', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # Plot top negative features (predicting Fake news)
    negative_features = [feature_names[i] for i in top_negative_idx]
    negative_weights = weights[top_negative_idx]
    
    ax2.barh(range(top_n), negative_weights, color='red', alpha=0.6)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(negative_features, fontsize=9)
    ax2.set_xlabel('Weight (Coefficient)', fontsize=11)
    ax2.set_title(f'Top {top_n} Features Predicting FAKE News', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved feature importance to: {save_path}")
    
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC (Receiver Operating Characteristic) curve.
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred_proba : array
        Predicted probabilities for positive class
    save_path : str or Path, optional
        Path to save the plot
    """
    # Calculate ROC curve points
    thresholds = np.linspace(1, 0, 100)
    tpr_values = []
    fpr_values = []
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    # Calculate AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(fpr_values)):
        auc += (fpr_values[i] - fpr_values[i-1]) * (tpr_values[i] + tpr_values[i-1]) / 2
    
    # Create plot
    plt.figure(figsize=(8, 8))
    
    # Plot ROC curve
    plt.plot(fpr_values, tpr_values, 'b-', linewidth=2.5, label=f'ROC Curve (AUC = {auc:.4f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Receiver Operating Characteristic', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add text box with interpretation
    textstr = f'AUC = {auc:.4f}\nExcellent' if auc > 0.9 else f'AUC = {auc:.4f}\nGood' if auc > 0.8 else f'AUC = {auc:.4f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    plt.text(0.6, 0.2, textstr, fontsize=12, bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved ROC curve to: {save_path}")
    
    plt.close()


def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot Precision-Recall curve.
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred_proba : array
        Predicted probabilities for positive class
    save_path : str or Path, optional
        Path to save the plot
    """
    # Calculate precision-recall curve points
    thresholds = np.linspace(1, 0, 100)
    precision_values = []
    recall_values = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precision_values.append(precision)
        recall_values.append(recall)
    
    # Calculate Average Precision (area under PR curve)
    avg_precision = np.mean(precision_values)
    
    # Create plot
    plt.figure(figsize=(8, 8))
    
    # Plot PR curve
    plt.plot(recall_values, precision_values, 'b-', linewidth=2.5, 
             label=f'PR Curve (Avg Precision = {avg_precision:.4f})')
    
    # Plot baseline (random classifier for balanced dataset)
    baseline = np.sum(y_true == 1) / len(y_true)
    plt.axhline(y=baseline, color='r', linestyle='--', linewidth=2, 
                label=f'Random Classifier (Baseline = {baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add text box
    textstr = f'Avg Precision = {avg_precision:.4f}'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    plt.text(0.6, 0.95, textstr, fontsize=12, bbox=props, 
             verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved Precision-Recall curve to: {save_path}")
    
    plt.close()


def plot_metrics_comparison(train_metrics, test_metrics, save_path=None):
    """
    Plot side-by-side comparison of training vs test metrics.
    
    Parameters:
    -----------
    train_metrics : dict
        Training set metrics
    test_metrics : dict
        Test set metrics
    save_path : str or Path, optional
        Path to save the plot
    """
    # Metrics to compare
    metric_names = ['Accuracy', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'F1-Score']
    metric_keys = ['accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'f1_score']
    
    train_values = [train_metrics[key] * 100 for key in metric_keys]
    test_values = [test_metrics[key] * 100 for key in metric_keys]
    
    # Create plot
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width/2, train_values, width, label='Training Set', 
                   color='skyblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, test_values, width, label='Test Set', 
                   color='lightcoral', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Training vs Test Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at 100%
    ax.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add interpretation text
    avg_gap = np.mean([train_values[i] - test_values[i] for i in range(len(metric_keys))])
    interpretation = "Excellent Generalization" if avg_gap < 1 else "Good Generalization" if avg_gap < 5 else "Some Overfitting"
    
    textstr = f'Avg Gap: {avg_gap:.2f}%\n{interpretation}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved metrics comparison to: {save_path}")
    
    plt.close()


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


def save_metrics_to_json(train_metrics, test_metrics, output_path):
    """
    Save training and test metrics to a JSON file.
    
    Parameters:
    -----------
    train_metrics : dict
        Training set metrics
    test_metrics : dict
        Test set metrics
    output_path : str or Path
        Path to save the JSON file
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        else:
            return obj
    
    metrics_data = {
        "training_set": convert_to_native(train_metrics),
        "test_set": convert_to_native(test_metrics)
    }
    
    with open(output_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    print(f"\nMetrics saved to: {output_path}")


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
    
    # Visualization save paths
    viz_dir = script_dir / "Visualizations_Out"
    viz_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
    
    confusion_matrix_path = viz_dir / "confusion_matrix.png"
    learning_curve_path = viz_dir / "learning_curve.png"
    feature_importance_path = viz_dir / "feature_importance.png"
    roc_curve_path = viz_dir / "roc_curve.png"
    pr_curve_path = viz_dir / "precision_recall_curve.png"
    metrics_comparison_path = viz_dir / "metrics_comparison.png"
    
    # Load configuration
    config = load_config(script_dir / 'config.json')
    hyperparams = config['hyperparameters']
    
    print(f"Using hyperparameters from config.json:")
    print(f"  Learning Rate:   {hyperparams['learning_rate']}")
    print(f"  Iterations:      {hyperparams['n_iterations']}")
    print(f"  Regularization:  {hyperparams['regularization']}")
    print()
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data(features_dir)
    
    # Train model with hyperparameters from config
    model = train_model(
        X_train, y_train,
        learning_rate=hyperparams['learning_rate'],
        n_iterations=hyperparams['n_iterations'],
        regularization=hyperparams['regularization']
    )
    
    # Evaluate model
    train_metrics, test_metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Save model
    save_model(model, model_save_path)
    
    # Save metrics to JSON
    metrics_json_path = script_dir / "train_metrics.json"
    save_metrics_to_json(train_metrics, test_metrics, metrics_json_path)
    
    # Get predicted probabilities for ROC and PR curves
    y_test_proba = model.predict_proba(X_test)
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\n[1/6] Creating confusion matrix...")
    plot_confusion_matrix(test_metrics, dataset_name='Test', save_path=confusion_matrix_path)
    
    print("\n[2/6] Creating learning curve...")
    plot_learning_curve(model, save_path=learning_curve_path)
    
    print("\n[3/6] Creating feature importance analysis...")
    plot_feature_importance(model, features_dir, top_n=20, save_path=feature_importance_path)
    
    print("\n[4/6] Creating ROC curve...")
    plot_roc_curve(y_test, y_test_proba, save_path=roc_curve_path)
    
    print("\n[5/6] Creating Precision-Recall curve...")
    plot_precision_recall_curve(y_test, y_test_proba, save_path=pr_curve_path)
    
    print("\n[6/6] Creating metrics comparison chart...")
    plot_metrics_comparison(train_metrics, test_metrics, save_path=metrics_comparison_path)
    
    print("\n[+] All visualizations saved to:", viz_dir)
    
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
    print("\nVisualizations saved:")
    print(f"  - {confusion_matrix_path.name}")
    print(f"  - {learning_curve_path.name}")
    print(f"  - {feature_importance_path.name}")
    print(f"  - {roc_curve_path.name}")
    print(f"  - {pr_curve_path.name}")
    print(f"  - {metrics_comparison_path.name}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

