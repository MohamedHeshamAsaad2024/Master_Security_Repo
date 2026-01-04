"""
Hyperparameter Tuning for Logistic Regression

This script performs grid search to find optimal hyperparameters
for the fake news detection model.
"""

import numpy as np
import json
import sys
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Data_preprocessing_and_cleanup"))

from logistic_regression import LogisticRegression
from utils import calculate_metrics
from features_pipeline import load_feature_matrices
import joblib


def load_config(config_path='config.json'):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config, config_path='config.json'):
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def create_validation_split(X_train, y_train, val_split=0.2, random_seed=42):
    """
    Split training data into train and validation sets.
    
    Parameters:
    -----------
    X_train : sparse matrix
        Training features
    y_train : array
        Training labels
    val_split : float
        Fraction of data to use for validation
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_tr, X_val, y_tr, y_val
    """
    np.random.seed(random_seed)
    n_samples = X_train.shape[0]
    indices = np.random.permutation(n_samples)
    
    n_val = int(n_samples * val_split)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    
    return X_train[train_idx], X_train[val_idx], y_train[train_idx], y_train[val_idx]


def train_and_evaluate(X_train, y_train, X_val, y_val, learning_rate, n_iterations, regularization):
    """
    Train model with given hyperparameters and evaluate on validation set.
    
    Returns:
    --------
    dict: validation metrics and model
    """
    # Train model
    model = LogisticRegression(
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        regularization=regularization
    )
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)
    
    metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
    
    return {
        'model': model,
        'metrics': metrics,
        'hyperparams': {
            'learning_rate': learning_rate,
            'n_iterations': n_iterations,
            'regularization': regularization
        }
    }


def grid_search(X_train, y_train, X_val, y_val, search_space):
    """
    Perform grid search over hyperparameter space.
    
    Parameters:
    -----------
    X_train, y_train : training data (after validation split)
    X_val, y_val : validation data
    search_space : dict
        Dictionary with lists of values for each hyperparameter
        
    Returns:
    --------
    list of results, best result
    """
    print("\n" + "="*70)
    print("GRID SEARCH - HYPERPARAMETER TUNING")
    print("="*70)
    
    # Generate all combinations
    param_names = list(search_space.keys())
    param_values = [search_space[name] for name in param_names]
    combinations = list(product(*param_values))
    
    print(f"\nTotal combinations to test: {len(combinations)}")
    print(f"Parameters: {param_names}")
    print()
    
    results = []
    best_result = None
    best_accuracy = 0
    
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))
        
        print(f"[{i}/{len(combinations)}] Testing: lr={params['learning_rate']}, "
              f"iter={params['n_iterations']}, reg={params['regularization']}")
        
        result = train_and_evaluate(
            X_train, y_train, X_val, y_val,
            learning_rate=params['learning_rate'],
            n_iterations=params['n_iterations'],
            regularization=params['regularization']
        )
        
        accuracy = result['metrics']['accuracy']
        print(f"         Validation Accuracy: {accuracy*100:.2f}%")
        
        results.append(result)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_result = result
    
    return results, best_result


def plot_hyperparameter_heatmaps(results, save_dir):
    """
    Create heatmaps showing hyperparameter effects.
    """
    # Extract data
    learning_rates = []
    regularizations = []
    n_iterations_list = []
    accuracies = []
    
    for result in results:
        hp = result['hyperparams']
        learning_rates.append(hp['learning_rate'])
        regularizations.append(hp['regularization'])
        n_iterations_list.append(hp['n_iterations'])
        accuracies.append(result['metrics']['accuracy'])
    
    # Get unique values
    unique_lr = sorted(set(learning_rates))
    unique_reg = sorted(set(regularizations))
    unique_iter = sorted(set(n_iterations_list))
    
    # Create heatmap for learning_rate vs regularization (averaged over iterations)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap 1: Learning Rate vs Regularization
    heatmap_data_1 = np.zeros((len(unique_reg), len(unique_lr)))
    count_1 = np.zeros((len(unique_reg), len(unique_lr)))
    
    for i, result in enumerate(results):
        hp = result['hyperparams']
        lr_idx = unique_lr.index(hp['learning_rate'])
        reg_idx = unique_reg.index(hp['regularization'])
        heatmap_data_1[reg_idx, lr_idx] += accuracies[i]
        count_1[reg_idx, lr_idx] += 1
    
    heatmap_data_1 = heatmap_data_1 / np.maximum(count_1, 1)
    
    sns.heatmap(heatmap_data_1, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=[f'{lr}' for lr in unique_lr],
                yticklabels=[f'{reg}' for reg in unique_reg],
                ax=axes[0], cbar_kws={'label': 'Accuracy'})
    axes[0].set_xlabel('Learning Rate', fontsize=12)
    axes[0].set_ylabel('Regularization (L2)', fontsize=12)
    axes[0].set_title('Learning Rate vs Regularization\n(Averaged over Iterations)', 
                      fontsize=13, fontweight='bold')
    
    # Heatmap 2: Learning Rate vs Iterations
    heatmap_data_2 = np.zeros((len(unique_iter), len(unique_lr)))
    count_2 = np.zeros((len(unique_iter), len(unique_lr)))
    
    for i, result in enumerate(results):
        hp = result['hyperparams']
        lr_idx = unique_lr.index(hp['learning_rate'])
        iter_idx = unique_iter.index(hp['n_iterations'])
        heatmap_data_2[iter_idx, lr_idx] += accuracies[i]
        count_2[iter_idx, lr_idx] += 1
    
    heatmap_data_2 = heatmap_data_2 / np.maximum(count_2, 1)
    
    sns.heatmap(heatmap_data_2, annot=True, fmt='.4f', cmap='YlGnBu',
                xticklabels=[f'{lr}' for lr in unique_lr],
                yticklabels=[f'{it}' for it in unique_iter],
                ax=axes[1], cbar_kws={'label': 'Accuracy'})
    axes[1].set_xlabel('Learning Rate', fontsize=12)
    axes[1].set_ylabel('Number of Iterations', fontsize=12)
    axes[1].set_title('Learning Rate vs Iterations\n(Averaged over Regularization)', 
                      fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    save_path = save_dir / 'hyperparameter_heatmaps.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmaps to: {save_path}")


def plot_top_configurations(results, top_n=10, save_dir=None):
    """
    Plot bar chart of top N configurations.
    """
    # Sort by accuracy
    sorted_results = sorted(results, key=lambda x: x['metrics']['accuracy'], reverse=True)
    top_results = sorted_results[:top_n]
    
    # Prepare data
    config_labels = []
    accuracies = []
    
    for i, result in enumerate(top_results, 1):
        hp = result['hyperparams']
        label = f"#{i}\nlr={hp['learning_rate']}\niter={hp['n_iterations']}\nreg={hp['regularization']}"
        config_labels.append(label)
        accuracies.append(result['metrics']['accuracy'] * 100)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    bars = ax.bar(range(top_n), accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.2f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(config_labels, fontsize=8)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Hyperparameter Configurations', fontsize=14, fontweight='bold')
    ax.set_ylim([min(accuracies) - 1, max(accuracies) + 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / f'top_{top_n}_configurations.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved top configurations plot to: {save_path}")
    else:
        plt.show()


def main():
    """Main hyperparameter tuning pipeline."""
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING - LOGISTIC REGRESSION")
    print("="*70)
    print()
    
    # Load configuration
    script_dir = Path(__file__).parent
    config = load_config(script_dir / 'config.json')
    
    print("Loaded configuration:")
    print(f"  Current hyperparameters: {config['hyperparameters']}")
    print(f"  Search space: {config['hyperparameter_search_space']}")
    print()
    
    # Setup paths
    features_dir = script_dir.parent / "Data_preprocessing_and_cleanup" / "Output" / "features_out"
    viz_dir = script_dir / config['paths']['visualizations_dir'] / 'Hyperparameter_Tuning'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    X_train_full, X_test, y_train_full, y_test = load_feature_matrices(str(features_dir), scaled=True)
    print(f"  Training set: {X_train_full.shape[0]:,} samples")
    print(f"  Test set: {X_test.shape[0]:,} samples")
    
    # Create validation split
    print(f"\nCreating validation split ({config['training']['validation_split']*100:.0f}% of training data)...")
    X_train, X_val, y_train, y_val = create_validation_split(
        X_train_full, y_train_full,
        val_split=config['training']['validation_split'],
        random_seed=config['training']['random_seed']
    )
    print(f"  New training set: {X_train.shape[0]:,} samples")
    print(f"  Validation set: {X_val.shape[0]:,} samples")
    
    # Perform grid search
    results, best_result = grid_search(
        X_train, y_train, X_val, y_val,
        config['hyperparameter_search_space']
    )
    
    # Display results
    print("\n" + "="*70)
    print("GRID SEARCH RESULTS")
    print("="*70)
    print(f"\nBest hyperparameters:")
    for param, value in best_result['hyperparams'].items():
        print(f"  {param}: {value}")
    
    print(f"\nBest validation metrics:")
    metrics = best_result['metrics']
    print(f"  Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"  Precision:   {metrics['precision']*100:.2f}%")
    print(f"  Recall:      {metrics['recall']*100:.2f}%")
    print(f"  F1-Score:    {metrics['f1_score']*100:.2f}%")
    if 'auc_roc' in metrics:
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    
    # Test on test set with best model
    print("\n" + "="*70)
    print("TESTING BEST MODEL ON TEST SET")
    print("="*70)
    
    best_model = best_result['model']
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
    
    print(f"\nTest set metrics:")
    print(f"  Accuracy:    {test_metrics['accuracy']*100:.2f}%")
    print(f"  Precision:   {test_metrics['precision']*100:.2f}%")
    print(f"  Recall:      {test_metrics['recall']*100:.2f}%")
    print(f"  F1-Score:    {test_metrics['f1_score']*100:.2f}%")
    if 'auc_roc' in test_metrics:
        print(f"  AUC-ROC:     {test_metrics['auc_roc']:.4f}")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\n[1/2] Creating hyperparameter heatmaps...")
    plot_hyperparameter_heatmaps(results, viz_dir)
    
    print("\n[2/2] Creating top configurations plot...")
    plot_top_configurations(results, top_n=10, save_dir=viz_dir)
    
    # Update config with best hyperparameters
    print("\n" + "="*70)
    print("UPDATING CONFIGURATION")
    print("="*70)
    
    response = input("\nDo you want to update config.json with best hyperparameters? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        config['hyperparameters'] = best_result['hyperparams']
        save_config(config, script_dir / 'config.json')
        print("✓ Configuration updated!")
        
        # Save best model
        model_path = script_dir / 'trained_fake_news_model_tuned.pkl'
        joblib.dump(best_model, model_path)
        print(f"✓ Best model saved to: {model_path}")
    else:
        print("Configuration not updated.")
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING COMPLETED!")
    print("="*70)
    print()


if __name__ == "__main__":
    main()
