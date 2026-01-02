
"""
Naive Bayes Classifier Training Script
======================================
Professional implementation of Naive Bayes algorithms for text classification.
This module includes mathematical implementations of:
1. Multinomial Naive Bayes (Frequency-based)
2. Complement Naive Bayes (Imbalanced dataset optimized)
3. Bernoulli Naive Bayes (Binary/Boolean based)

It performs exhaustive Hyperparameter Optimization (Grid Search) with K-Fold 
Cross-Validation to select the optimal model architecture and parameters.
"""

import sys
import os
import argparse
import numpy as np
import json
from time import time
import joblib
from scipy import sparse

# ---------------------------------------------------------
# 1. Import Feature Pipeline
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
preprocessing_dir = os.path.abspath(os.path.join(current_dir, '..', 'Data_preprocessing_and_cleanup'))
if preprocessing_dir not in sys.path:
    sys.path.append(preprocessing_dir)

try:
    from features_pipeline import load_feature_matrices
except ImportError:
    print(f"Error: Could not import 'features_pipeline' from {preprocessing_dir}")
    sys.exit(1)

# ---------------------------------------------------------
# Custom Naive Bayes Implementation
# ---------------------------------------------------------
try:
    from naive_bayes_model import MultinomialNB, ComplementNB, BernoulliNB
except ImportError:
    # If running from different directory, try adding current dir to path
    sys.path.append(current_dir)
    from naive_bayes_model import MultinomialNB, ComplementNB, BernoulliNB

# ---------------------------------------------------------
# Metrics & Helper Functions
# ---------------------------------------------------------
def get_metrics(y_true, y_pred):
    # Classes: 0 (Fake), 1 (Real)
    # We want metrics for each, or macro avg.
    # Let's compute global macro F1 as the optimization target
    
    unique_labels = np.unique(y_true)
    
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    
    for c in unique_labels:
        # One-vs-Rest
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        precision_sum += p
        recall_sum += r
        f1_sum += f1
        
    n = len(unique_labels)
    
    acc = np.sum(y_true == y_pred) / len(y_true)
    
    return {
        "accuracy": acc,
        "precision": precision_sum / n,
        "recall": recall_sum / n,
        "f1_macro": f1_sum / n
    }

def print_confusion_matrix(y_true, y_pred, labels=[0, 1]):
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    return cm

# ---------------------------------------------------------
# Hyperparameter Tuning (Custom GridSearch)
# ---------------------------------------------------------
class BayesOptimizer:
    def __init__(self, model_class, param_grid):
        self.model_class = model_class
        self.param_grid = param_grid
        
    def _generate_params(self):
        # Cartesian product of params
        from itertools import product
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        for instance in product(*values):
            yield dict(zip(keys, instance))

    def run_cv(self, X, y, k=5):
        best_f1 = -1
        best_params = None
        best_metrics = None
        best_model_ref = None
        
        param_list = list(self._generate_params())
        print(f"    Evaluating {len(param_list)} parameter combinations...")
        
        # Shuffle indices for CV
        indices = np.arange(X.shape[0])
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = len(indices) // k
        
        for params in param_list:
            # We will average all metrics across folds
            metric_accum = {
                "accuracy": [], "precision": [], "recall": [], "f1_macro": []
            }
            
            # K-Fold Loop
            for i in range(k):
                val_idx = indices[i*fold_size : (i+1)*fold_size]
                train_idx = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
                
                X_tr_fold, y_tr_fold = X[train_idx], y[train_idx]
                X_val_fold, y_val_fold = X[val_idx], y[val_idx]
                
                model = self.model_class(**params)
                model.fit(X_tr_fold, y_tr_fold)
                y_pred_fold = model.predict(X_val_fold)
                
                m = get_metrics(y_val_fold, y_pred_fold)
                for key in metric_accum:
                    metric_accum[key].append(m[key])
            
            # Calculate averages
            avg_metrics = {k: np.mean(v) for k, v in metric_accum.items()}
            
            # Calculate Holistic Composite Score (Average of all 4 key metrics)
            composite = (avg_metrics['f1_macro'] + avg_metrics['recall'] + 
                         avg_metrics['accuracy'] + avg_metrics['precision']) / 4.0
            
            if composite > best_f1:
                best_f1 = composite
                best_params = params
                best_metrics = avg_metrics
                
        # Refit best on full train set
        best_model_ref = self.model_class(**best_params)
        best_model_ref.fit(X, y)
        
        return best_model_ref, best_params, best_metrics

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Naive Bayes Models")
    parser.add_argument("--features_dir", type=str, 
                        default=os.path.join(preprocessing_dir, "Output", "features_out"),
                        help="Path to feature pipeline output directory")
    parser.add_argument("--output_dir", type=str, 
                        default=os.path.join(current_dir, "models"), 
                        help="Directory to save trained models")
    parser.add_argument("--mnb_grid", type=str, default=None, help="JSON string for MNB grid")
    parser.add_argument("--cnb_grid", type=str, default=None, help="JSON string for CNB grid")
    parser.add_argument("--bnb_grid", type=str, default=None, help="JSON string for BNB grid")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n[1] Loading Data...")
    try:
        X_train, X_test, y_train, y_test = load_feature_matrices(args.features_dir, scaled=False)
        print(f"    Train: {X_train.shape}, Test: {X_test.shape}")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Define Search Spaces (Override with CLI if provided)
    def parse_grid(json_str, default):
        if not json_str: return default
        try:
            return json.loads(json_str)
        except:
            print(f"Warning: Failed to parse grid '{json_str}'. Using default.")
            return default

    search_spaces = [
        {
            "name": "MultinomialNB",
            "class": MultinomialNB,
            "grid": parse_grid(args.mnb_grid, {"alpha": [0.01, 0.1, 1.0], "fit_prior": [True, False]})
        },
        {
            "name": "ComplementNB",
            "class": ComplementNB,
            "grid": parse_grid(args.cnb_grid, {"alpha": [0.01, 0.1, 1.0], "fit_prior": [True, False], "norm": [False, True]})
        },
        {
            "name": "BernoulliNB",
            "class": BernoulliNB,
            "grid": parse_grid(args.bnb_grid, {"alpha": [0.01, 0.1, 1.0], "fit_prior": [True, False], "binarize": [0.0]})
        }
    ]

    candidate_results = []

    print("\n[2] Starting Hyperparameter Optimization (5-Fold CV)...")
    
    for space in search_spaces:
        print(f"\n--- Tuning {space['name']} ---")
        optimizer = BayesOptimizer(space["class"], space["grid"])
        # Now run_cv returns metrics dict as 3rd arg
        model, params, metrics = optimizer.run_cv(X_train, y_train, k=5)
        
        print(f"    Best Params: {params}")
        print(f"    Best CV F1: {metrics['f1_macro']:.4f}")
        
        candidate_results.append({
            "name": space["name"],
            "model": model,
            "params": params,
            "metrics": metrics
        })

    # Compare candidates
    print("\n[3] Model Comparison (Best Config per Algorithm):")
    print("-" * 88)
    print(f"{'Model':<15} | {'F1 Score':<10} | {'Recall':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Composite':<10}")
    print("-" * 88)
    
    best_overall_model = None
    best_overall_score = -1
    best_overall_name = ""
    
    for res in candidate_results:
        m = res['metrics']
        # Calculate Holistic Composite Score (Average of all 4 key metrics)
        composite = (m['f1_macro'] + m['recall'] + m['accuracy'] + m['precision']) / 4.0
        
        print(f"{res['name']:<15} | {m['f1_macro']:<10.4f} | {m['recall']:<10.4f} | {m['accuracy']:<10.4f} | {m['precision']:<10.4f} | {composite:<10.4f}")
        
        # Selection Strategy: Maximize Holistic Composite Score
        if composite > best_overall_score:
            best_overall_score = composite
            best_overall_model = res['model']
            best_overall_name = res['name']
    
    print("-" * 88)
    print(f"Winner: {best_overall_name} (Holistic Score: {best_overall_score:.4f})")


    
    # Evaluate Winner on Test Set
    print("\n[4] Final Evaluation on Test Set...")
    y_pred = best_overall_model.predict(X_test)
    metrics = get_metrics(y_test, y_pred)
    
    print("    Test Set Metrics:")
    print(f"    - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    - Precision: {metrics['precision']:.4f}")
    print(f"    - Recall:    {metrics['recall']:.4f}")
    print(f"    - F1 Score:  {metrics['f1_macro']:.4f}")
    
    cm = print_confusion_matrix(y_test, y_pred)
    print(f"\n    Confusion Matrix:\n{cm}")

    # 5. Save Model Variants
    print("\n[5] Saving Model Variants...")
    for res in candidate_results:
        algo_name = res["name"].lower() # e.g., multinomialnb
        # Map to shorter key if desired: mnb, cnb, bnb
        key = algo_name.replace("naivebayes", "") 
        # Actually res["name"] is MultinomialNB, etc.
        if "Multinomial" in res["name"]: key = "mnb"
        elif "Complement" in res["name"]: key = "cnb"
        elif "Bernoulli" in res["name"]: key = "bnb"
        
        save_path = os.path.join(args.output_dir, f"{key}.joblib")
        joblib.dump(res["model"], save_path)
        print(f"    - {res['name']} saved to: {save_path}")

    # Also save the "best" one as default
    best_path = os.path.join(args.output_dir, "best_naive_bayes.joblib")
    joblib.dump(best_overall_model, best_path)
    print(f"    - Default Best ({best_overall_name}) saved to: {best_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
