"""
Test Trained Model on WELFake External Dataset

This script demonstrates how to use the feature_pipeline APIs to test
the trained logistic regression model on an external dataset (WELFake).

Uses:
- load_artifacts() to load the preprocessing pipeline
- transform_records() to preprocess new data
- The trained model to make predictions
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import json

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Data_preprocessing_and_cleanup"))

from features_pipeline import load_artifacts, transform_records
from utils import calculate_metrics, print_metrics
import joblib

# Import visualization functions from train_fake_news
from train_fake_news import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_metrics_comparison,
    plot_feature_importance
)


def load_config(config_path='config.json'):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {config_path} not found.")
        return None


def load_welfake_data(csv_path, limit=None):
    """
    Load WELFake dataset.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to WELFake_Dataset.csv
    limit : int, optional
        Limit number of samples (for testing)
        
    Returns:
    --------
    titles, texts, labels : lists
    """
    print("Loading WELFake dataset...")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Apply limit if specified
    if limit:
        df = df.head(limit)
        print(f"  Limited to {limit} samples for testing")
    
    print(f"  Total samples: {len(df):,}")
    print(f"  Fake news: {(df['label'] == 0).sum():,}")
    print(f"  Real news: {(df['label'] == 1).sum():,}")
    
    # Extract data
    titles = df['title'].fillna('').tolist()
    texts = df['text'].fillna('').tolist()
    labels = df['label'].values
    
    return titles, texts, labels


def preprocess_welfake_data(titles, texts, artifacts, scaled=True):
    """
    Preprocess WELFake data using the feature_pipeline API.
    
    This demonstrates using transform_records() to apply the same
    preprocessing that was used during training.
    
    Parameters:
    -----------
    titles : list
        Article titles
    texts : list
        Article texts
    artifacts : dict
        Loaded artifacts from load_artifacts()
    scaled : bool
        Whether to scale features (True for LR/SVM)
        
    Returns:
    --------
    X : sparse matrix
        Preprocessed features ready for prediction
    """
    from features_pipeline import FeatureConfig
    
    print("\nPreprocessing WELFake data using feature_pipeline API...")
    print("  Using transform_records() to apply same preprocessing as training")
    
    # Create config matching what was used during training
    # (include_subject=False was used for training)
    cfg = FeatureConfig(include_subject=False)
    
    # Use transform_records API (as documented in feature_pipeline.md lines 110-122)
    # subjects=None because WELFake doesn't have subject column
    X = transform_records(
        titles=titles,
        texts=texts,
        subjects=None,  # WELFake doesn't have subject column
        artifacts=artifacts,
        config=cfg,
        scaled=scaled
    )
    
    print(f"  Preprocessed features shape: {X.shape}")
    print(f"  Features are {'scaled' if scaled else 'unscaled'}")
    
    return X


def evaluate_on_welfake(model, X, y):
    """
    Evaluate the model on WELFake dataset.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained model
    X : sparse matrix
        Preprocessed features
    y : array
        True labels
        
    Returns:
    --------
    metrics : dict
        Evaluation metrics
    """
    print("\n" + "="*70)
    print("EVALUATING ON WELFAKE EXTERNAL DATASET")
    print("="*70)
    
    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Calculate metrics
    metrics = calculate_metrics(y, y_pred, y_pred_proba)
    print_metrics(metrics)
    
    return metrics


def main():
    """
    Main testing pipeline for WELFake dataset.
    """
    print("\n" + "="*70)
    print("TESTING TRAINED MODEL ON WELFAKE EXTERNAL DATASET")
    print("="*70)
    print()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    # Paths to artifacts and model
    features_dir = project_dir / "Data_preprocessing_and_cleanup" / "Output" / "features_out"
    model_path = script_dir / "trained_fake_news_model.pkl"
    
    # Path to WELFake dataset
    welfake_path = project_dir / "Data_preprocessing_and_cleanup" / "External_Datasets" / "WELFake_Dataset.csv"
    
    # Visualization save paths
    viz_dir = script_dir / "Visualizations_Out" / "WELFake_Test"
    viz_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    
    confusion_matrix_path = viz_dir / "confusion_matrix_welfake.png"
    roc_curve_path = viz_dir / "roc_curve_welfake.png"
    pr_curve_path = viz_dir / "precision_recall_curve_welfake.png"
    feature_importance_path = viz_dir / "feature_importance_welfake.png"
    
    # Check if files exist
    if not welfake_path.exists():
        print(f"ERROR: WELFake dataset not found at {welfake_path}")
        return
    
    if not model_path.exists():
        print(f"ERROR: Trained model not found at {model_path}")
        return
    
    # Load configuration to display hyperparameters used
    config = load_config(script_dir / 'config.json')
    if config:
        print("="*70)
        print("MODEL CONFIGURATION")
        print("="*70)
        print("\nHyperparameters used for training:")
        hyperparams = config.get('hyperparameters', {})
        print(f"  Learning Rate:   {hyperparams.get('learning_rate', 'N/A')}")
        print(f"  Iterations:      {hyperparams.get('n_iterations', 'N/A')}")
        print(f"  Regularization:  {hyperparams.get('regularization', 'N/A')}")
        print()
    
    # Step 1: Load artifacts using feature_pipeline API (API #3)
    print("="*70)
    print("STEP 1: LOADING PREPROCESSING ARTIFACTS")
    print("="*70)
    print()
    
    artifacts = load_artifacts(str(features_dir))
    print("[+] Loaded artifacts:")
    for key in artifacts.keys():
        print(f"  - {key}")
    print()
    
    # Step 2: Load trained model
    print("="*70)
    print("STEP 2: LOADING TRAINED MODEL")
    print("="*70)
    print()
    
    model = joblib.load(model_path)
    print(f"[+] Loaded model from: {model_path.name}")
    print()
    
    # Step 3: Load WELFake data
    print("="*70)
    print("STEP 3: LOADING WELFAKE DATA")
    print("="*70)
    print()
    
    # Use limit=10000 for faster testing (remove or increase for full dataset)
    titles, texts, labels = load_welfake_data(welfake_path, limit=10000)
    print()
    
    # Step 4: Preprocess using feature_pipeline API (API #4)
    print("="*70)
    print("STEP 4: PREPROCESSING WITH FEATURE_PIPELINE API")
    print("="*70)
    print()
    
    X = preprocess_welfake_data(titles, texts, artifacts, scaled=True)
    print()
    
    # Step 5: Evaluate model
    print("="*70)
    print("STEP 5: EVALUATION")
    print("="*70)
    
    metrics = evaluate_on_welfake(model, X, labels)
    
    # Get predictions and probabilities for visualizations
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Step 6: Generate visualizations
    print("\n" + "="*70)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\n[1/4] Creating confusion matrix...")
    plot_confusion_matrix(metrics, dataset_name='WELFake', save_path=confusion_matrix_path)
    
    print("\n[2/4] Creating ROC curve...")
    plot_roc_curve(labels, y_pred_proba, save_path=roc_curve_path)
    
    print("\n[3/4] Creating Precision-Recall curve...")
    plot_precision_recall_curve(labels, y_pred_proba, save_path=pr_curve_path)
    
    print("\n[4/4] Creating feature importance analysis...")
    plot_feature_importance(model, features_dir, top_n=20, save_path=feature_importance_path)
    
    print("\n[+] All visualizations saved to:", viz_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("CROSS-DATASET EVALUATION COMPLETED!")
    print("="*70)
    print()
    print("Model trained on: ISOT Dataset")
    print("Model tested on:  WELFake Dataset")
    print()
    print("Final Results:")
    print(f"  Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"  Precision:   {metrics['precision']*100:.2f}%")
    print(f"  Recall:      {metrics['recall']*100:.2f}%")
    print(f"  Sensitivity: {metrics['sensitivity']*100:.2f}%")
    print(f"  Specificity: {metrics['specificity']*100:.2f}%")
    print(f"  F1-Score:    {metrics['f1_score']*100:.2f}%")
    if 'auc_roc' in metrics:
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    print()
    print("Visualizations saved:")
    print(f"  - {confusion_matrix_path.name}")
    print(f"  - {roc_curve_path.name}")
    print(f"  - {pr_curve_path.name}")
    print(f"  - {feature_importance_path.name}")
    print()
    print("="*70)
    print()


if __name__ == "__main__":
    main()

