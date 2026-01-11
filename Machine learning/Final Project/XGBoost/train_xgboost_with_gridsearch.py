"""
================================================================================
Module Name: xgboost_train_advanced.py
================================================================================

Description:
    Enhanced XGBoost Training Module with GridSearchCV and Hyperparameter Analysis.
    
    Features:
    - GridSearchCV for hyperparameter optimization
    - Individual plots for each hyperparameter's effect
    - Learning curve visualization
    - Feature importance analysis
    - Comprehensive model evaluation

================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from joblib import dump, load


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_PREPROCESSING_DIR = PROJECT_ROOT / "Data_preprocessing_and_cleanup"
sys.path.insert(0, str(DATA_PREPROCESSING_DIR))
import features_pipeline


# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================

OUTPUT_MODEL_DIR = CURRENT_DIR / "XgbTrainedModel"
OUTPUT_PLOTS_DIR = CURRENT_DIR / "XgbTrainedOutput"
FEATURES_DIR = DATA_PREPROCESSING_DIR / "Output" / "features_out"

# ------------------------------------------------------------------------------
# GRID SEARCH PARAMETERS
# ------------------------------------------------------------------------------

PARAM_GRID = {
    'n_estimators': [90],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1],
    'subsample': [0.6],
    'gamma': [0.1,1,2],
    'reg_alpha': [0.01],
    'reg_lambda': [0.1,1,2]
}

RANDOM_STATE = 42
 #In order to reduce overfitting in hyperparameter optimization, we decreased the values of parameters such as max_depth,n_estimators and increased the gamma,lambda values.
# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_training_data(features_dir=None, use_scaled=True):
    """Load training and test data"""
    if features_dir is None:
        features_dir = FEATURES_DIR

    print("=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)

    # Assuming features_pipeline has these functions
    X_train, X_test, y_train, y_test = features_pipeline.load_feature_matrices(
        out_dir=features_dir,
        scaled=use_scaled
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print("=" * 60)

    return X_train, X_test, y_train, y_test

# ==============================================================================
# GRID SEARCH TRAINING
# ==============================================================================

def train_with_gridsearch(X_train, y_train):
    """Train XGBoost using GridSearchCV"""
    
    print("=" * 60)
    print("GRID SEARCH HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # Create base model
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        cv=5,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1,
        return_train_score=True
    )
    
    # Train with grid search
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\nGrid search completed in {training_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    print("=" * 60)
    
    return grid_search

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_hyperparameter_analysis(grid_search, output_dir):
    """Create plots for each hyperparameter's effect"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert CV results to DataFrame
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    # 1. Plot for each hyperparameter
    for param in PARAM_GRID.keys():
        if param in cv_results.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Group by parameter value
            param_df = cv_results.groupby(f'param_{param}')['mean_test_score'].agg(['mean', 'std'])
            param_df = param_df.reset_index()
            
            # Plot
            ax.errorbar(param_df[f'param_{param}'], param_df['mean'], 
                       yerr=param_df['std'], fmt='o-', capsize=5, linewidth=2)
            
            ax.set_xlabel(param.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel('Mean CV Accuracy', fontsize=12)
            ax.set_title(f'Effect of {param.replace("_", " ").title()} on Model Performance', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'param_analysis_{param}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    # 2. Heatmap of parameter combinations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Select key parameter pairs for heatmaps
    param_pairs = [
        ('max_depth', 'learning_rate'),
        ('n_estimators', 'learning_rate'),
        ('subsample', 'colsample_bytree'),
        ('gamma', 'reg_lambda')
    ]
    
    for idx, (param1, param2) in enumerate(param_pairs):
        if idx < len(axes):
            ax = axes[idx]
            
            if f'param_{param1}' in cv_results.columns and f'param_{param2}' in cv_results.columns:
                # Create pivot table
                pivot_table = cv_results.pivot_table(
                    values='mean_test_score',
                    index=f'param_{param1}',
                    columns=f'param_{param2}',
                    aggfunc='mean'
                )
                
                # Plot heatmap
                im = ax.imshow(pivot_table.values, cmap='viridis', aspect='auto')
                
                # Set labels
                ax.set_xticks(range(len(pivot_table.columns)))
                ax.set_yticks(range(len(pivot_table.index)))
                ax.set_xticklabels([f'{x:.2f}' if isinstance(x, float) else str(x) 
                                   for x in pivot_table.columns])
                ax.set_yticklabels([str(x) for x in pivot_table.index])
                
                ax.set_xlabel(param2.replace('_', ' ').title(), fontsize=10)
                ax.set_ylabel(param1.replace('_', ' ').title(), fontsize=10)
                ax.set_title(f'{param1} vs {param2}', fontsize=12, fontweight='bold')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, label='Mean CV Accuracy')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Hyperparameter analysis plots saved")

def plot_learning_curves(model, X_train, y_train, output_dir):
    """Plot learning curves"""
    
    output_dir = Path(output_dir)
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score', linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                    alpha=0.1, color='blue')
    
    ax.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-validation score', linewidth=2)
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                    alpha=0.1, color='green')
    
    ax.set_xlabel('Training Examples', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Learning Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Learning curves plot saved")

def plot_feature_importance(model, X_train, output_dir):
    """Plot feature importance"""
    
    output_dir = Path(output_dir)
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Get top 20 features
    top_n = min(20, len(importance))
    indices = np.argsort(importance)[-top_n:][::-1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(range(top_n), importance[indices])
    ax.set_yticks(range(top_n))
    
    # Try to get feature names, otherwise use indices
    try:
        # Assuming features might have names from the pipeline
        feature_names = [f"Feature {i}" for i in indices]
        ax.set_yticklabels(feature_names)
    except:
        ax.set_yticklabels([f"Feature {i}" for i in indices])
    
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Feature importance plot saved")

def plot_model_evaluation(model, X_test, y_test, output_dir):
    """Create comprehensive evaluation plots"""
    
    output_dir = Path(output_dir)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'], 
                ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve', fontweight='bold')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Prediction Distribution
    axes[1, 0].hist(y_pred_proba, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Predicted Probability (Real News)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Prediction Probability Distribution', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Metrics Comparison
    report = classification_report(y_test, y_pred, target_names=['Fake', 'Real'], output_dict=True)
    
    metrics = ['Accuracy', 'Precision\n(Fake)', 'Recall\n(Fake)', 
               'Precision\n(Real)', 'Recall\n(Real)']
    values = [
        accuracy_score(y_test, y_pred),
        report['Fake']['precision'],
        report['Fake']['recall'],
        report['Real']['precision'],
        report['Real']['recall']
    ]
    
    colors = ['blue', 'red', 'red', 'green', 'green']
    bars = axes[1, 1].bar(metrics, values, color=colors, edgecolor='black')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Metrics', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('XGBoost Model Evaluation', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Model evaluation plots saved")

# ==============================================================================
# EVALUATION FUNCTION
# ==============================================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=["Fake News", "Real News"]
        )
    }
    
    print(metrics["classification_report"])
    return metrics

# ==============================================================================
# MODEL SAVING
# ==============================================================================

def save_model_and_results(model, grid_search, metrics, output_dir):
    """Save model and all results"""
    
    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"xgboost_gridsearch_{timestamp}"
    
    # Save model
    model_path = OUTPUT_MODEL_DIR / f"{model_name}.joblib"
    dump(model, model_path)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "metrics": {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1_score"])
        },
        "cv_results": {
            "mean_test_score": float(grid_search.cv_results_['mean_test_score'][grid_search.best_index_]),
            "std_test_score": float(grid_search.cv_results_['std_test_score'][grid_search.best_index_]),
            "mean_fit_time": float(grid_search.cv_results_['mean_fit_time'][grid_search.best_index_])
        }
    }
    
    with open(OUTPUT_MODEL_DIR / f"{model_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return str(model_path), metadata

# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def train_xgboost_with_gridsearch():
    """Main training pipeline"""
    
    # Load data
    X_train, X_test, y_train, y_test = load_training_data()
    
    # Train with GridSearchCV
    grid_search = train_with_gridsearch(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    metrics = evaluate_model(best_model, X_test, y_test)
    
    # Create all plots
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_hyperparameter_analysis(grid_search, OUTPUT_PLOTS_DIR)
    plot_learning_curves(best_model, X_train, y_train, OUTPUT_PLOTS_DIR)
    plot_feature_importance(best_model, X_train, OUTPUT_PLOTS_DIR)
    plot_model_evaluation(best_model, X_test, y_test, OUTPUT_PLOTS_DIR)
    
    # Save model and results
    model_path, metadata = save_model_and_results(best_model, grid_search, metrics, OUTPUT_MODEL_DIR)
    
    # Create summary
    results = {
        "model": best_model,
        "grid_search": grid_search,
        "metrics": metrics,
        "model_path": model_path,
        "metadata": metadata,
        "plots_dir": str(OUTPUT_PLOTS_DIR)
    }
    
    return results

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("XGBOOST WITH GRIDSEARCH - FAKE NEWS DETECTION")
    print("=" * 70)
    
    results = train_xgboost_with_gridsearch()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"Best Parameters: {results['grid_search'].best_params_}")
    print(f"Best CV Score: {results['grid_search'].best_score_:.4f}")
    print(f"Test Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Model Saved: {results['model_path']}")
    print(f"Plots Saved: {results['plots_dir']}")
    print("=" * 70)