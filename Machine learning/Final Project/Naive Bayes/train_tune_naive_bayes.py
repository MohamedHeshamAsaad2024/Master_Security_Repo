
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
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump, load

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
    # 3. Define Pipeline & Parameter Grid
    # ---------------------------------------------------------
    print("\n[2] Setting up GridSearchCV")

    # We use a Pipeline to include Feature Selection in the tuning process
    pipeline = Pipeline([
        ('selector', SelectKBest(score_func=chi2)),
        ('clf', MultinomialNB()) # Placeholder class, will be replaced by grid search
    ])

    # Define Parameter Grid
    # Note: 'clf' can be switched out for different NB variants
    param_grid = [
        # Variant 1: MultinomialNB
        {
            'selector__k': ['all', 5000, 10000],          # Feature selection tuning
            'clf': [MultinomialNB()],
            'clf__alpha': [0.01, 0.1, 0.5, 1.0, 5.0],     # Smoothing parameters
            'clf__fit_prior': [True, False]
        },
        # Variant 2: ComplementNB (Often better for imbalanced datasets, though ISOT is balanced)
        {
            'selector__k': ['all', 5000, 10000],
            'clf': [ComplementNB()],
            'clf__alpha': [0.01, 0.1, 0.5, 1.0, 5.0],
            'clf__fit_prior': [True, False],
            'clf__norm': [True, False]
        },
        # Variant 3: BernoulliNB (Uses binary occurrence only)
        # Note: BernoulliNB might perform worse with TF-IDF values, usually expects binary.
        # However, scikit-learn's BernoulliNB implementation handles non-binary by binarizing internally if 'binarize' set.
        {
            'selector__k': ['all', 10000],
            'clf': [BernoulliNB()],
            'clf__alpha': [0.01, 0.1, 1.0],
            'clf__binarize': [0.0, 0.1] # Threshold for binarizing TF-IDF
        }
    ]

    # ---------------------------------------------------------
    # 4. Perform Grid Search
    # ---------------------------------------------------------
    print("    Starting Hyperparameter Tuning... (This may take a moment)")
    t0 = time()
    
    # We use 5-fold Cross Validation
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='f1_macro', # optimizing for F1 score (balanced precision/recall)
        n_jobs=-1,          # Use all CPUs
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    
    print(f"    Done in {time() - t0:.2f}s")
    print("\n    Best Parameters Found:")
    pprint(grid_search.best_params_)
    print(f"    Best CV Score (F1 Macro): {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

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
    clf_step = best_model.named_steps['clf']
    selector_step = best_model.named_steps['selector']
    
    # We need the vocabulary to map indices to words
    artifacts = load_artifacts(args.features_dir)
    tfidf = artifacts['tfidf']
    feature_names = np.array(tfidf.get_feature_names_out())
    
    # Using SelectKBest masks feature names if k != 'all'
    if selector_step.get_support() is not None:
        kept_features_mask = selector_step.get_support()
        input_feature_names = feature_names[kept_features_mask]
        # Also need subject features if included?
        # The artifact extraction handles 'tfidf' words. Feature pipeline hstacks subject at the end.
        # For simplicity, we'll focus on text feature mapping here.
        # If 'include_subject' was true, feature_names would be shorter than X.columns.
        # But 'tfidf.get_feature_names_out()' only gives text features.
        
        # NOTE: If include_subject=True, feature_pipeline adds subject columns AFTER text.
        # We need to handle that if we want robust name mapping. 
        # But `feature_names` from tfidf is only partial if subject is there.
        # For now, we assume simple text mapping or handle length mismatch gently.
        pass
    else:
        input_feature_names = feature_names

    # Check for feature_log_prob_ (Multinomial/Complement/Bernoulli)
    if hasattr(clf_step, 'feature_log_prob_'):
        print("    Top 10 Indicative Features for 'Fake' News (Class 0):")
        # specific to binary classification: log_prob is shape (2, n_features)
        # Class 0 is first row. Higher log prob = more associated with class.
        
        # IMPORTANT: feature_log_prob_ is P(x_i|y), but for 'predictive' words we often want 
        # to see largest difference between classes or just highest prob.
        # Let's show highest prob for Class 0
        
        class_0_probs = clf_step.feature_log_prob_[0]
        # Get top indices
        top_indices = class_0_probs.argsort()[-10:][::-1]
        
        # Safety check for dimensions
        if len(input_feature_names) == clf_step.feature_log_prob_.shape[1]:
            for idx in top_indices:
                print(f"      {input_feature_names[idx]}: {class_0_probs[idx]:.4f}")
        else:
            print("      (Cannot map feature names directly due to shape mismatch - likely Subject OHE features added or Selector mismatch)")
            
        print("\n    Top 10 Indicative Features for 'Real' News (Class 1):")
        class_1_probs = clf_step.feature_log_prob_[1]
        top_indices = class_1_probs.argsort()[-10:][::-1]
        
        if len(input_feature_names) == clf_step.feature_log_prob_.shape[1]:
            for idx in top_indices:
                print(f"      {input_feature_names[idx]}: {class_1_probs[idx]:.4f}")

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
