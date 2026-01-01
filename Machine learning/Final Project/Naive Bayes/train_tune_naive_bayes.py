
"""
Naive Bayes Classifier Training and Tuning Script
=================================================

This script performs the following steps:
1.  Loads preprocessed feature matrices (unscaled) from the Data Preprocessing module.
2.  Performs Hyperparameter Optimization using GridSearchCV.
    -   Models: MultinomialNB, ComplementNB, BernoulliNB
    -   Params: alpha, fit_prior, norm
    -   Feature Selection: SelectKBest (tuning k)
3.  Evaluates the best model on the Test Set.
    -   Accuracy, Precision, Recall, F1
    -   Confusion Matrix
4.  Demonstrates GUI Integration:
    -   Extracts and prints 'Weighted' features (most indicative words).
    -   Tests 'predict' function with raw text input.
5.  Saves the best trained model for deployment.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import json
from time import time
from pprint import pprint

# ---------------------------------------------------------
# 1. Import Feature Pipeline (Sibling Directory)
# ---------------------------------------------------------
# Add preprocessing directory to sys.path to allow importing feature_pipeline
current_dir = os.path.dirname(os.path.abspath(__file__))
preprocessing_dir = os.path.abspath(os.path.join(current_dir, '..', 'Data preprocessing and cleanup'))
if preprocessing_dir not in sys.path:
    sys.path.append(preprocessing_dir)

try:
    from features_pipeline import load_feature_matrices, load_artifacts, transform_records, FeatureConfig
except ImportError:
    try:
        # Fallback if the file is named feature_pipeline.py
        from feature_pipeline import load_feature_matrices, load_artifacts, transform_records, FeatureConfig
    except ImportError:
        print(f"Error: Could not import 'features_pipeline' or 'feature_pipeline' from {preprocessing_dir}")
        print("Please ensure the 'Data preprocessing and cleanup' folder exists and is a sibling of 'Naive Bayes'.")
        sys.exit(1)

# ---------------------------------------------------------
# Imports for Modeling
# ---------------------------------------------------------
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump, load

def tune_model(model, param_grid, X_train, y_train):
    """
    Performs GridSearchCV to find the best hyperparameters for a given model.
    """
    print(f"\n--- Tuning {type(model).__name__} ---")
    
    # Grid Search
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=5, 
        scoring='f1_macro', 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_score_

def main():
    parser = argparse.ArgumentParser(description="Train and Tune Naive Bayes Model")
    parser.add_argument("--features_dir", type=str, 
                        default=os.path.join(preprocessing_dir, "Output", "features_out"),
                        help="Path to feature pipeline output directory")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save trained models")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print(" NAIVE BAYES TRAINING & TUNING ")
    print("="*60)

    # ---------------------------------------------------------
    # 2. Load Data
    # ---------------------------------------------------------
    print(f"\n[1] Loading Feature Matrices from: {args.features_dir}")
    try:
        # Load UNSCALED matrices (Naive Bayes works with counts/tf-idf directly, scaling not needed/damaging)
        X_train, X_test, y_train, y_test = load_feature_matrices(args.features_dir, scaled=False)
        print(f"    Train Shape: {X_train.shape}, Labels: {y_train.shape}")
        print(f"    Test Shape:  {X_test.shape}, Labels: {y_test.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # ---------------------------------------------------------
    # 3. Define Parameter Grids for each Classifier
    # ---------------------------------------------------------
    print("\n[2] Setting up GridSearchCV for individual Naive Bayes models")

    # Define Parameter Grid for MultinomialNB
    mnb_param_grid = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 5.0],     # Smoothing parameters
        'fit_prior': [True, False]
    }

    # Define Parameter Grid for ComplementNB
    cnb_param_grid = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 5.0],
        'fit_prior': [True, False],
        'norm': [True, False]
    }

    # Define Parameter Grid for BernoulliNB
    bnb_param_grid = {
        'alpha': [0.01, 0.1, 1.0],
        'binarize': [0.0, 0.1] # Threshold for binarizing TF-IDF
    }

    # ---------------------------------------------------------
    # 4. Perform Grid Search for each model
    # ---------------------------------------------------------
    print("    Starting Hyperparameter Tuning... (This may take a moment)")
    t0 = time()
    
    best_overall_model = None
    best_overall_score = -np.inf
    
    # Tune MultinomialNB
    mnb_model, mnb_score = tune_model(MultinomialNB(), mnb_param_grid, X_train, y_train)
    if mnb_score > best_overall_score:
        best_overall_score = mnb_score
        best_overall_model = mnb_model

    # Tune ComplementNB
    cnb_model, cnb_score = tune_model(ComplementNB(), cnb_param_grid, X_train, y_train)
    if cnb_score > best_overall_score:
        best_overall_score = cnb_score
        best_overall_model = cnb_model

    # Tune BernoulliNB
    bnb_model, bnb_score = tune_model(BernoulliNB(), bnb_param_grid, X_train, y_train)
    if bnb_score > best_overall_score:
        best_overall_score = bnb_score
        best_overall_model = bnb_model

    print(f"\n    Done in {time() - t0:.2f}s")
    print("\n    Overall Best Model Found:")
    print(f"    Type: {type(best_overall_model).__name__}")
    print(f"    Best CV Score (F1 Macro): {best_overall_score:.4f}")

    best_model = best_overall_model

    # ---------------------------------------------------------
    # 5. Evaluate on Test Set
    # ---------------------------------------------------------
    print("\n[3] Evaluating on Test Set")
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    # class 0 = Fake, class 1 = Real
    target_names = ['Fake', 'Real']
    
    print(f"    Accuracy: {acc:.4f}")
    print("\n    Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("    Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # ---------------------------------------------------------
    # 6. Save Model
    # ---------------------------------------------------------
    print("\n[4] Saving Model")
    model_path = os.path.join(args.output_dir, "best_naive_bayes.joblib")
    dump(best_model, model_path)
    print(f"    Model saved to: {model_path}")

    # ---------------------------------------------------------
    # 7. GUI Integration Helpers
    # ---------------------------------------------------------
    print("\n[5] GUI Integration Verification")

    # A. Feature Importance (Weights)
    # Extract the classifier step
    
    # We need the vocabulary to map indices to words
    artifacts = load_artifacts(args.features_dir)
    
    # Check for feature_log_prob_ (Multinomial/Complement/Bernoulli)
    if hasattr(best_model, 'feature_log_prob_'):
        # Extract feature log probabilities
        # Since we are using the model directly, we access it directly
        
        # Get feature names from the saved artifact
        # Note: We need to use the full 20k vocabulary since we didn't do selection
        # But the logic below assumes we have the right mapping
        
        # Since we removed SelectKBest, the model weights match the original feature set (20k)
        # So we can map directly to 'feature_names'
        
        feature_names = artifacts["tfidf"].get_feature_names_out()
        feature_log_probs = best_model.feature_log_prob_ # Shape (2, n_features)
        
        # Calculate "informativeness" - difference between classes or absolute weight
        # For Naive Bayes, feature_log_prob_ is log P(x_i|y)
        # Class 0: Fake, Class 1: Real (Assuming label encoding)
        
        fake_probs = feature_log_probs[0]
        real_probs = feature_log_probs[1]
        
        # To find words indicative of "Fake", we want P(w|Fake) >> P(w|Real)
        # Or simply highest P(w|Fake)
        
        print("\n--- Top Indicative Features (Training Set) ---")
        
        top_n = 10
        
        # Fake Indicators
        fake_indices = np.argsort(fake_probs)[-top_n:][::-1]
        print("\nTop words for FAKE NEWS:")
        for idx in fake_indices:
            print(f"  - {feature_names[idx]}: {fake_probs[idx]:.4f}")

        # Real Indicators
        real_indices = np.argsort(real_probs)[-top_n:][::-1]
        print("\nTop words for REAL NEWS:")
        for idx in real_indices:
            print(f"  - {feature_names[idx]}: {real_probs[idx]:.4f}")

    # B. Test Prediction Function
    print("\n    Testing Inference Function (transform_records -> predict):")
    sample_title = "Breaking: Alien spaceship lands in Time Square"
    sample_text = "NASA confirmed today that a UFO has landed in New York City. The aliens are asking for pizza."
    
    print(f"      Input Title: {sample_title}")
    
    # 1. Transform raw text using existing artifacts
    config_path = os.path.join(args.features_dir, "artifacts", "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
         config_dict = json.load(f)
    cfg = FeatureConfig(**config_dict)
    
    # CAUTION: If Training used SelectKBest, we MUST apply it during inference too.
    # The 'best_model' is a Pipeline that INCLUDES the selector.
    # So we just pass the transformed matrix (full features) to the pipeline.
    
    try:
        # transform_records returns the FULL feature set (matching what was generated during build)
        X_sample = transform_records(
            titles=[sample_title], 
            texts=[sample_text], 
            subjects=None, 
            artifacts=artifacts, 
            config=cfg, 
            scaled=False 
        )
        
        # Predict using the fitted pipeline (which handles selection + classification)
        prediction = best_model.predict(X_sample)[0]
        label = "Real" if prediction == 1 else "Fake"
        print(f"      Prediction: {label} (Class {prediction})")
        
        # Get probability if available
        if hasattr(best_model, "predict_proba"):
            probs = best_model.predict_proba(X_sample)[0]
            print(f"      Confidence: Fake={probs[0]:.4f}, Real={probs[1]:.4f}")
            
    except Exception as e:
        print(f"      Inference failed: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
