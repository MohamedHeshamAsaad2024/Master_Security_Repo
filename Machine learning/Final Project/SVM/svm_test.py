"""
================================================================================
Module Name: svm_test.py
================================================================================

Description:
    Support Vector Machine (SVM) Testing Module for Fake News Classification.
    
    This module provides testing functionality with:
    - Live progress bar during testing
    - Real-time updates for ALL metrics (accuracy, precision, recall, F1)
    - Testing on external datasets (WELFake)
    - Detailed output and visualization
    
Dependencies:
    - svm_train.py (for model loading and shared APIs)
    - features_pipeline.py (for data transformation)
    - tqdm (for progress bar)

================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# Standard library imports
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# Add current directory and parent to path for imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_PREPROCESSING_DIR = PROJECT_ROOT / "Data_preprocessing_and_cleanup"
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(DATA_PREPROCESSING_DIR))

# Import from svm_train module (shared APIs)
from svm_train import (
    load_trained_model,
    predict_single,
    predict_batch,
    OUTPUT_MODEL_DIR,
    FEATURES_DIR,
    USE_SCALED_FEATURES,
    plot_roc_curve
)

# Import from features_pipeline
from features_pipeline import (
    load_artifacts,
    FeatureConfig,
    load_welfake_external_eval
)


# ==============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ==============================================================================
# 
# This section contains all configurable parameters for the SVM testing module.
# Each configuration includes:
#   - DESCRIPTION: What this configuration controls
#   - AFFECTED STEPS: Which parts of the pipeline use this configuration
#   - CALLED BY: Which function(s) use this configuration
#   - POSSIBLE VALUES: Options you can test
#   - EXPECTED BEHAVIOR: What happens with different values
#   - DEPENDENCIES: Other configurations that affect or are affected by this one
# ==============================================================================


# ------------------------------------------------------------------------------
# OUTPUT PATH CONFIGURATIONS
# ------------------------------------------------------------------------------

# CONFIGURATION: OUTPUT_TEST_DIR
# DESCRIPTION: Directory path where test results, plots, and reports will be saved.
#              Includes confusion matrix, metrics report, and prediction results.
# AFFECTED STEPS: Evaluation step, Save Results step
# CALLED BY: save_test_results(), test_on_csv()
# POSSIBLE VALUES: Any valid directory path (relative or absolute)
# EXPECTED BEHAVIOR:
#   - Directory will be created if it doesn't exist
#   - Results saved with timestamps to avoid overwriting
# DEPENDENCIES: None
OUTPUT_TEST_DIR = CURRENT_DIR / "SvmTestOutput"


# CONFIGURATION: DEFAULT_MODEL_PATH
# DESCRIPTION: Default path to the trained SVM model to use for testing.
#              Can be overridden by passing model_path parameter to functions.
# AFFECTED STEPS: Load Model step
# CALLED BY: load_model_for_testing(), test_on_csv()
# POSSIBLE VALUES: 
#   - Path to specific .joblib model file
#   - None (will search for latest model in OUTPUT_MODEL_DIR)
# EXPECTED BEHAVIOR:
#   - If None, will find the most recent model in the models directory
#   - If specified, will load that exact model
# DEPENDENCIES: Model must exist and be compatible with current features
DEFAULT_MODEL_PATH = None


# CONFIGURATION: WELFAKE_DATASET_PATH
# DESCRIPTION: Path to the WELFake external dataset for cross-dataset evaluation.
#              This is a completely unseen dataset to test generalization.
#              The dataset is loaded and transformed via load_welfake_external_eval() API
#              from features_pipeline.py, which handles preprocessing.
# AFFECTED STEPS: Load Test Data step
# CALLED BY: test_on_welfake()
# POSSIBLE VALUES: Path to WELFake_Dataset.csv
# EXPECTED BEHAVIOR:
#   - Dataset must have columns: title, text, label
#   - Labels: 0=fake, 1=real
# DEPENDENCIES: Requires pre-trained model and feature artifacts from features_pipeline
WELFAKE_DATASET_PATH = DATA_PREPROCESSING_DIR / "External_Datasets" / "WELFake_Dataset.csv"


# ------------------------------------------------------------------------------
# TESTING CONFIGURATION
# ------------------------------------------------------------------------------

# CONFIGURATION: BATCH_SIZE
# DESCRIPTION: Number of samples to process at once during batch testing.
#              Larger batches are faster but use more memory.
# AFFECTED STEPS: Testing step, Live Updates step
# CALLED BY: run_test_on_features(), test_on_csv()
# POSSIBLE VALUES AND EXPECTED BEHAVIOR:
#   - 100: Frequent updates, more overhead, lower memory
#   - 500: Balanced (RECOMMENDED)
#   - 1000: Faster processing, less frequent updates, more memory
#   - 5000: Fast but infrequent updates
# DEPENDENCIES: Affects how often live metrics are updated
BATCH_SIZE = 500


# CONFIGURATION: LIMIT_SAMPLES
# DESCRIPTION: Limit the number of samples to test.
#              Useful for quick testing or when memory is limited.
# AFFECTED STEPS: Load Test Data step
# CALLED BY: test_on_csv(), test_on_welfake()
# POSSIBLE VALUES AND EXPECTED BEHAVIOR:
#   - None: Test all samples in the dataset
#   - 1000: Quick test on first 1000 samples
#   - 10000: Moderate test
#   - 50000: Larger test
# DEPENDENCIES: None
LIMIT_SAMPLES = None


# CONFIGURATION: SHOW_PROGRESS_BAR
# DESCRIPTION: Whether to display a live progress bar in the terminal.
#              Uses tqdm library for smooth progress visualization.
# AFFECTED STEPS: Testing step
# CALLED BY: run_test_on_features(), test_on_csv()
# POSSIBLE VALUES:
#   - True: Show progress bar with ETA and speed
#   - False: No progress bar (useful for logging to file)
# EXPECTED BEHAVIOR:
#   - True: Visual progress bar updates in terminal
#   - False: Silent processing
# DEPENDENCIES: None
SHOW_PROGRESS_BAR = True


# CONFIGURATION: SHOW_LIVE_METRICS
# DESCRIPTION: Whether to display live metric updates during testing.
#              Shows accuracy, precision, recall, F1 after each batch.
# AFFECTED STEPS: Testing step, Live Updates step
# CALLED BY: run_test_on_features(), test_on_csv()
# POSSIBLE VALUES:
#   - True: Show metrics updating in real-time
#   - False: Only show final metrics at the end
# EXPECTED BEHAVIOR:
#   - True: Metrics displayed and updated after each batch
#   - False: Faster testing, fewer console updates
# DEPENDENCIES: Requires SHOW_PROGRESS_BAR=True for inline updates
SHOW_LIVE_METRICS = True


# CONFIGURATION: SAVE_PREDICTIONS
# DESCRIPTION: Whether to save individual predictions to a CSV file.
#              Useful for error analysis and debugging.
# AFFECTED STEPS: Save Results step
# CALLED BY: save_test_results(), test_on_csv()
# POSSIBLE VALUES:
#   - True: Save predictions CSV with title, text, true label, predicted label
#   - False: Only save aggregate metrics
# EXPECTED BEHAVIOR:
#   - True: Creates predictions CSV (can be large for big datasets)
#   - False: Smaller output, faster completion
# DEPENDENCIES: None
SAVE_PREDICTIONS = True


# ==============================================================================
# API FUNCTIONS
# ==============================================================================


def find_latest_model(model_dir: str = None) -> str:
    """
    ===========================================================================
    FIND LATEST MODEL
    ===========================================================================
    
    Description:
        Finds the most recently created model file in the models directory.
    
    Parameters:
        model_dir : str, optional
            Directory to search for models.
            Default: OUTPUT_MODEL_DIR from svm_train.py
    
    Returns:
        str : Path to the latest model file.
    
    Raises:
        FileNotFoundError: If no models are found.
    
    Example:
        >>> model_path = find_latest_model()
        >>> print(f"Latest model: {model_path}")
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Resolve model directory
    # -------------------------------------------------------------------------
    
    if model_dir is None:
        model_dir = OUTPUT_MODEL_DIR
    
    model_path = Path(model_dir)
    
    # -------------------------------------------------------------------------
    # Step 2: Check if directory exists
    # -------------------------------------------------------------------------
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            f"Please train a model first using svm_train.py"
        )
    
    # -------------------------------------------------------------------------
    # Step 3: Find all model files
    # -------------------------------------------------------------------------
    
    # List all .joblib files in the directory
    model_files = list(model_path.glob("*.joblib"))
    
    # -------------------------------------------------------------------------
    # Step 4: Check if any models exist
    # -------------------------------------------------------------------------
    
    if len(model_files) == 0:
        raise FileNotFoundError(
            f"No model files found in: {model_path}\n"
            f"Please train a model first using svm_train.py"
        )
    
    # -------------------------------------------------------------------------
    # Step 5: Sort by modification time and get latest
    # -------------------------------------------------------------------------
    
    # Sort files by modification time (newest first)
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Get the most recent file
    latest_model = model_files[0]
    
    print(f"Found latest model: {latest_model.name}")
    
    return str(latest_model)


def load_model_for_testing(model_path: str = None) -> object:
    """
    ===========================================================================
    LOAD MODEL FOR TESTING
    ===========================================================================
    
    Description:
        Loads a trained SVM model for testing.
        If no path specified, finds the latest model.
    
    Parameters:
        model_path : str, optional
            Path to the model file.
            Default: Finds latest model in OUTPUT_MODEL_DIR
    
    Returns:
        SVC : Loaded SVM model.
    
    Example:
        >>> model = load_model_for_testing()
        >>> print(f"Model loaded: {model.kernel}")
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Resolve model path
    # -------------------------------------------------------------------------
    
    if model_path is None:
        # Check global configuration
        if DEFAULT_MODEL_PATH is not None:
            model_path = DEFAULT_MODEL_PATH
        else:
            # Find the latest model
            model_path = find_latest_model()
    
    # -------------------------------------------------------------------------
    # Step 2: Load and return the model
    # -------------------------------------------------------------------------
    
    print("=" * 60)
    print("LOADING MODEL FOR TESTING")
    print("=" * 60)
    print(f"Model path: {model_path}")
    
    model = load_trained_model(model_path)
    
    print(f"Model kernel: {model.kernel}")
    print(f"Model C parameter: {model.C}")
    print("=" * 60)
    
    return model


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None
) -> dict:
    """
    ===========================================================================
    CALCULATE METRICS
    ===========================================================================
    
    Description:
        Calculates ALL classification metrics including Specificity and AUC.
    
    Parameters:
        y_true : numpy.ndarray
            True labels.
        
        y_pred : numpy.ndarray
            Predicted labels.
            
        y_score : numpy.ndarray, optional
            Confidence scores or probabilities (for AUC).
    
    Returns:
        dict : Dictionary containing all metrics:
            - accuracy : float
            - precision : float
            - recall : float (Sensitivity)
            - specificity : float
            - f1_score : float
            - auc_score : float
            - confusion_matrix : numpy.ndarray
    
    Equations:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP)
        Recall (Sensitivity) = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
        AUC = Area Under ROC Curve
    
    Example:
        >>> metrics = calculate_metrics(y_true, y_pred, y_score)
        >>> print(f"AUC: {metrics['auc_score']:.4f}")
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Calculate accuracy
    # -------------------------------------------------------------------------
    
    # Accuracy = (TP + TN) / Total
    accuracy = accuracy_score(y_true, y_pred)
    
    # -------------------------------------------------------------------------
    # Step 2: Calculate precision
    # -------------------------------------------------------------------------
    
    # Precision = TP / (TP + FP)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # -------------------------------------------------------------------------
    # Step 3: Calculate Recall (Sensitivity)
    # -------------------------------------------------------------------------
    
    # Recall (Sensitivity) = TP / (TP + FN)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # -------------------------------------------------------------------------
    # Step 4: Calculate F1-score
    # -------------------------------------------------------------------------
    
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # -------------------------------------------------------------------------
    # Step 5: Calculate confusion matrix and Specificity
    # -------------------------------------------------------------------------
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Calculate Specificity: TN / (TN + FP)
    if conf_matrix.shape == (2, 2):
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
        
    # -------------------------------------------------------------------------
    # Step 6: Calculate AUC (if scores provided)
    # -------------------------------------------------------------------------
    
    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = 0.0
    else:
        auc = 0.0
    
    # -------------------------------------------------------------------------
    # Step 7: Return all metrics
    # -------------------------------------------------------------------------
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'auc_score': auc,
        'confusion_matrix': conf_matrix
    }
    
    return metrics


def format_metrics_string(metrics: dict) -> str:
    """
    ===========================================================================
    FORMAT METRICS STRING
    ===========================================================================
    
    Description:
        Creates a formatted string showing all metrics.
        Used for live display during testing.
    
    Parameters:
        metrics : dict
            Metrics dictionary from calculate_metrics().
    
    Returns:
        str : Formatted string for display.
    
    Example:
        >>> metrics_str = format_metrics_string(metrics)
        >>> print(metrics_str)
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Format each metric
    # -------------------------------------------------------------------------
    
    formatted = (
        f"Acc: {metrics['accuracy']*100:.2f}% | "
        f"Prec: {metrics['precision']*100:.2f}% | "
        f"Rec(Sens): {metrics['recall']*100:.2f}% | "
        f"Spec: {metrics['specificity']*100:.2f}% | "
        f"AUC: {metrics['auc_score']:.4f} | "
        f"F1: {metrics['f1_score']*100:.2f}%"
    )
    
    return formatted





def run_test_on_features(
    model: object,
    X: np.ndarray,
    y_true: np.ndarray,
    batch_size: int = None,
    show_progress: bool = None,
    show_live_metrics: bool = None
) -> dict:
    """
    ===========================================================================
    RUN TEST ON PRE-TRANSFORMED FEATURES
    ===========================================================================
    
    Description:
        Runs testing with live progress bar and real-time metric updates.
        This function works directly with pre-transformed feature matrices,
        avoiding redundant feature extraction. Used with load_welfake_external_eval().
    
    Parameters:
        model : SVC
            Trained SVM model.
        
        X : numpy.ndarray or sparse matrix
            Pre-transformed feature matrix (from load_welfake_external_eval).
        
        y_true : numpy.ndarray
            True labels for evaluation.
        
        batch_size : int, optional
            Number of samples per batch.
            Default: BATCH_SIZE global configuration.
        
        show_progress : bool, optional
            Whether to show progress bar.
            Default: SHOW_PROGRESS_BAR global configuration.
        
        show_live_metrics : bool, optional
            Whether to show live metric updates.
            Default: SHOW_LIVE_METRICS global configuration.
    
    Returns:
        dict : Test results containing:
            - predictions : numpy.ndarray
            - metrics : dict
            - processing_time : float
    
    Example:
        >>> X_wel, y_wel = load_welfake_external_eval(path, features_dir)
        >>> results = run_test_on_features(model, X_wel, y_wel)
        >>> print(f"Final accuracy: {results['metrics']['accuracy']:.4f}")
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Resolve configuration
    # -------------------------------------------------------------------------
    
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    if show_progress is None:
        show_progress = SHOW_PROGRESS_BAR
    
    if show_live_metrics is None:
        show_live_metrics = SHOW_LIVE_METRICS
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize
    # -------------------------------------------------------------------------
    
    n_samples = X.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Initialize predictions and scores arrays
    all_predictions = np.zeros(n_samples, dtype=int)
    all_scores = np.zeros(n_samples, dtype=float)
    
    # Track which samples have been processed
    processed = 0
    
    print("\n" + "=" * 60)
    print("RUNNING TEST ON PRE-TRANSFORMED FEATURES")
    print("=" * 60)
    print(f"Total samples: {n_samples}")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {n_batches}")
    print("-" * 60)
    
    # -------------------------------------------------------------------------
    # Step 3: Record start time
    # -------------------------------------------------------------------------
    
    start_time = time.time()
    
    # -------------------------------------------------------------------------
    # Step 4: Create progress bar
    # -------------------------------------------------------------------------
    
    if show_progress:
        progress_bar = tqdm(
            total=n_samples,
            desc="Testing",
            unit="samples",
            ncols=150
        )
    
    # -------------------------------------------------------------------------
    # Step 5: Process batches
    # -------------------------------------------------------------------------
    
    for batch_idx in range(n_batches):
        # Calculate batch boundaries
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_count = end_idx - start_idx
        
        # Get batch features (already transformed)
        X_batch = X[start_idx:end_idx]
        
        # Make predictions
        batch_predictions = model.predict(X_batch)
        
        # Get scores for AUC
        if hasattr(model, "decision_function"):
            batch_scores = model.decision_function(X_batch)
        elif hasattr(model, "predict_proba"):
            batch_scores = model.predict_proba(X_batch)[:, 1]
        else:
            # Fallback
            batch_scores = batch_predictions
            
        # Store predictions and scores
        all_predictions[start_idx:end_idx] = batch_predictions
        all_scores[start_idx:end_idx] = batch_scores
        
        # Update processed count
        processed = end_idx
        
        # Update progress bar
        if show_progress:
            progress_bar.update(batch_count)
        
        # Show live metrics if enabled
        if show_live_metrics:
            # Calculate current metrics (on processed samples so far)
            current_metrics = calculate_metrics(
                y_true[:processed],
                all_predictions[:processed],
                all_scores[:processed]
            )
            
            # Format and display
            metrics_str = format_metrics_string(current_metrics)
            
            if show_progress:
                progress_bar.set_postfix_str(metrics_str)
            else:
                print(f"\rProcessed: {processed}/{n_samples} | {metrics_str}", end="")
    
    # -------------------------------------------------------------------------
    # Step 6: Close progress bar
    # -------------------------------------------------------------------------
    
    if show_progress:
        progress_bar.close()
    else:
        print()  # New line after live updates
    
    # -------------------------------------------------------------------------
    # Step 7: Calculate final metrics
    # -------------------------------------------------------------------------
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    final_metrics = calculate_metrics(y_true, all_predictions, all_scores)
    
    # -------------------------------------------------------------------------
    # Step 8: Print final results
    # -------------------------------------------------------------------------
    
    print("\n" + "-" * 60)
    print("FINAL RESULTS")
    print("-" * 60)
    print(f"Accuracy:             {final_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:            {final_metrics['precision'] * 100:.2f}%")
    print(f"Recall (Sensitivity): {final_metrics['recall'] * 100:.2f}%")
    print(f"Specificity:          {final_metrics['specificity'] * 100:.2f}%")
    print(f"ROC AUC:              {final_metrics['auc_score']:.4f}")
    print(f"F1-Score:             {final_metrics['f1_score'] * 100:.2f}%")
    print(f"\nProcessing time: {processing_time:.2f} seconds")
    print(f"Speed: {n_samples / processing_time:.2f} samples/second")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 9: Return results
    # -------------------------------------------------------------------------
    
    results = {
        'predictions': all_predictions,
        'scores': all_scores,
        'metrics': final_metrics,
        'processing_time': processing_time
    }
    
    return results


def save_test_results(
    results: dict,
    titles: list = None,
    texts: list = None,
    y_true: np.ndarray = None,
    output_dir: str = None,
    test_name: str = None,
    save_predictions: bool = None
) -> dict:
    """
    ===========================================================================
    SAVE TEST RESULTS
    ===========================================================================
    
    Description:
        Saves test results, including metrics, plots, and optionally predictions.
    
    Parameters:
        results : dict
            Test results from run_test_on_features().
        
        titles : list, optional
            Original titles (for saving predictions).
        
        texts : list, optional
            Original texts (for saving predictions).
        
        y_true : numpy.ndarray, optional
            True labels (for saving predictions).
        
        output_dir : str, optional
            Output directory.
            Default: OUTPUT_TEST_DIR global configuration.
        
        test_name : str, optional
            Name for the test run.
            Default: "test_<timestamp>"
        
        save_predictions : bool, optional
            Whether to save individual predictions.
            Default: SAVE_PREDICTIONS global configuration.
    
    Returns:
        dict : Paths to saved files.
    
    Example:
        >>> paths = save_test_results(results, titles, texts, y_true)
        >>> print(f"Metrics saved to: {paths['metrics']}")
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Resolve configuration
    # -------------------------------------------------------------------------
    
    if output_dir is None:
        output_dir = OUTPUT_TEST_DIR
    
    if save_predictions is None:
        save_predictions = SAVE_PREDICTIONS
    
    if test_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = f"test_{timestamp}"
    
    # -------------------------------------------------------------------------
    # Step 2: Create output directory
    # -------------------------------------------------------------------------
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("SAVING TEST RESULTS")
    print("=" * 60)
    
    saved_paths = {}
    
    # -------------------------------------------------------------------------
    # Step 3: Save metrics JSON
    # -------------------------------------------------------------------------
    
    metrics_data = {
        'test_name': test_name,
        'timestamp': datetime.now().isoformat(),
        'accuracy': float(results['metrics']['accuracy']),
        'precision': float(results['metrics']['precision']),
        'recall': float(results['metrics']['recall']),
        'specificity': float(results['metrics']['specificity']),
        'f1_score': float(results['metrics']['f1_score']),
        'auc_score': float(results['metrics']['auc_score']),
        'confusion_matrix': results['metrics']['confusion_matrix'].tolist(),
        'processing_time': results['processing_time'],
        'n_samples': len(results['predictions'])
    }
    
    metrics_path = output_path / f"{test_name}_metrics.json"
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"Metrics saved to: {metrics_path}")
    saved_paths['metrics'] = str(metrics_path)
    
    # -------------------------------------------------------------------------
    # Step 4: Save confusion matrix plot
    # -------------------------------------------------------------------------
    
    conf_matrix = results['metrics']['confusion_matrix']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    classes = ['Fake News', 'Real News']
    ax.set(
        xticks=np.arange(conf_matrix.shape[1]),
        yticks=np.arange(conf_matrix.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f'Test Confusion Matrix - {test_name}',
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    # Add annotations
    thresh = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(
                j, i,
                format(conf_matrix[i, j], 'd'),
                ha="center", va="center",
                color="white" if conf_matrix[i, j] > thresh else "black"
            )
    
    fig.tight_layout()
    
    cm_path = output_path / f"{test_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Confusion matrix saved to: {cm_path}")
    saved_paths['confusion_matrix'] = str(cm_path)
    
    # -------------------------------------------------------------------------
    # Step 5: Save ROC Curve
    # -------------------------------------------------------------------------
    
    if y_true is not None and 'scores' in results:
        roc_path = plot_roc_curve(
            y_true=y_true,
            y_score=results['scores'],
            auc_score_val=results['metrics']['auc_score'],
            output_dir=output_dir,
            filename=f"{test_name}_roc_curve.png"
        )
        saved_paths['roc_curve'] = roc_path
    
    # -------------------------------------------------------------------------
    # Step 6: Save predictions CSV if requested
    # -------------------------------------------------------------------------
    
    if save_predictions and titles is not None and texts is not None and y_true is not None:
        predictions_df = pd.DataFrame({
            'title': titles,
            'text': [t[:200] + '...' if len(t) > 200 else t for t in texts],  # Truncate
            'true_label': y_true,
            'predicted_label': results['predictions'],
            'true_class': ['Fake' if l == 0 else 'Real' for l in y_true],
            'predicted_class': ['Fake' if p == 0 else 'Real' for p in results['predictions']],
            'correct': y_true == results['predictions']
        })
        
        predictions_path = output_path / f"{test_name}_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        
        print(f"Predictions saved to: {predictions_path}")
        saved_paths['predictions'] = str(predictions_path)
    
    # -------------------------------------------------------------------------
    # Step 6: Save classification report
    # -------------------------------------------------------------------------
    
    report = classification_report(
        y_true if y_true is not None else np.zeros(len(results['predictions'])),
        results['predictions'],
        target_names=['Fake News', 'Real News']
    )
    
    report_path = output_path / f"{test_name}_classification_report.txt"
    
    with open(report_path, 'w') as f:
        f.write(f"Test Results: {test_name}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        f.write("METRICS SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:             {results['metrics']['accuracy'] * 100:.2f}%\n")
        f.write(f"Precision:            {results['metrics']['precision'] * 100:.2f}%\n")
        f.write(f"Recall (Sensitivity): {results['metrics']['recall'] * 100:.2f}%\n")
        f.write(f"Specificity:          {results['metrics']['specificity'] * 100:.2f}%\n")
        f.write(f"ROC AUC:              {results['metrics']['auc_score']:.4f}\n")
        f.write(f"F1-Score:             {results['metrics']['f1_score'] * 100:.2f}%\n")
        f.write("\n\nCLASSIFICATION REPORT\n")
        f.write("-" * 40 + "\n")
        f.write(report)
    
    print(f"Report saved to: {report_path}")
    saved_paths['report'] = str(report_path)
    
    print("=" * 60)
    
    return saved_paths


def test_on_csv(
    csv_path: str,
    model_path: str = None,
    output_dir: str = None,
    batch_size: int = None,
    limit: int = None,
    save_predictions: bool = None
) -> dict:
    """
    ===========================================================================
    TEST ON CSV
    ===========================================================================
    
    Description:
        Loads a CSV dataset and runs the complete test pipeline.
        CSV must have columns: title, text, label
    
    Parameters:
        csv_path : str
            Path to the CSV file to test on.
        
        model_path : str, optional
            Path to the model file.
            Default: Latest model in OUTPUT_MODEL_DIR
        
        output_dir : str, optional
            Output directory for results.
            Default: OUTPUT_TEST_DIR
        
        batch_size : int, optional
            Batch size for processing.
            Default: BATCH_SIZE
        
        limit : int, optional
            Limit number of samples.
            Default: LIMIT_SAMPLES
        
        save_predictions : bool, optional
            Save individual predictions.
            Default: SAVE_PREDICTIONS
    
    Returns:
        dict : Complete test results with paths.
    
    Example:
        >>> results = test_on_csv("data/test.csv")
        >>> print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Print header
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 70)
    print("                    SVM TESTING PIPELINE")
    print("=" * 70)
    print(f"Dataset: {csv_path}")
    
    # -------------------------------------------------------------------------
    # Step 2: Resolve configuration
    # -------------------------------------------------------------------------
    
    if limit is None:
        limit = LIMIT_SAMPLES
    
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    # -------------------------------------------------------------------------
    # Step 3: Load the model
    # -------------------------------------------------------------------------
    
    model = load_model_for_testing(model_path)
    
    # -------------------------------------------------------------------------
    # Step 4: Load and transform dataset using API
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 60)
    print("LOADING TEST DATA VIA FEATURES_PIPELINE API")
    print("=" * 60)
    print(f"CSV path: {csv_path}")
    
    # Load and transform using the Data_preprocessing_and_cleanup API
    # This efficiently handles loading, cleaning, vectorization, and scaling
    X, y = load_welfake_external_eval(
        welfake_csv_path=csv_path,
        features_out_dir=FEATURES_DIR,
        scaled=USE_SCALED_FEATURES,
        limit=limit
    )
    
    # If we need to save predictions, we need the original titles and texts.
    # The API returns only matrices, so we read the CSV just to get metadata.
    titles = None
    texts = None
    y_true = y  # Use the labels returned by the API
    
    if save_predictions:
        print("Loading original text for prediction saving...")
        df_meta = pd.read_csv(csv_path)
        if limit is not None:
            df_meta = df_meta.head(limit)
        
        titles = df_meta['title'].fillna('').astype(str).tolist()
        texts = df_meta['text'].fillna('').astype(str).tolist()
    
    print(f"Total samples: {X.shape[0]}")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Class distribution: fake={np.sum(y == 0)}, real={np.sum(y == 1)}")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 5: Run test on features
    # -------------------------------------------------------------------------
    
    results = run_test_on_features(
        model=model,
        X=X,
        y_true=y,
        batch_size=batch_size
    )
    
    # -------------------------------------------------------------------------
    # Step 6: Save results
    # -------------------------------------------------------------------------
    
    # Generate test name from CSV filename
    csv_name = Path(csv_path).stem
    test_name = f"test_{csv_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    saved_paths = save_test_results(
        results=results,
        titles=titles,
        texts=texts,
        y_true=y_true,
        output_dir=output_dir,
        test_name=test_name,
        save_predictions=save_predictions
    )
    
    # -------------------------------------------------------------------------
    # Step 7: Return complete results
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 70)
    print("                    TESTING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nFinal Accuracy:       {results['metrics']['accuracy'] * 100:.2f}%")
    print(f"Final Precision:      {results['metrics']['precision'] * 100:.2f}%")
    print(f"Final Recall (Sens):  {results['metrics']['recall'] * 100:.2f}%")
    print(f"Final Specificity:    {results['metrics']['specificity'] * 100:.2f}%")
    print(f"Final ROC AUC:        {results['metrics']['auc_score']:.4f}")
    print(f"Final F1-Score:       {results['metrics']['f1_score'] * 100:.2f}%")
    print(f"Results saved to:     {output_dir if output_dir else OUTPUT_TEST_DIR}")
    print("=" * 70)
    
    complete_results = {
        'metrics': results['metrics'],
        'predictions': results['predictions'],
        'processing_time': results['processing_time'],
        'saved_paths': saved_paths
    }
    
    return complete_results


def test_on_welfake(
    model_path: str = None,
    limit: int = None,
    output_dir: str = None
) -> dict:
    """
    ===========================================================================
    TEST ON WELFAKE
    ===========================================================================
    
    Description:
        Tests the model on the WELFake external dataset.
        This is a cross-dataset evaluation to test generalization.
        
        Uses load_welfake_external_eval() API from features_pipeline.py
        to load and transform the dataset, avoiding redundant processing.
    
    Parameters:
        model_path : str, optional
            Path to the model file.
            Default: Latest model in OUTPUT_MODEL_DIR
        
        limit : int, optional
            Limit number of samples.
            Default: LIMIT_SAMPLES
        
        output_dir : str, optional
            Output directory for results.
            Default: OUTPUT_TEST_DIR
    
    Returns:
        dict : Test results.
    
    Example:
        >>> results = test_on_welfake(limit=10000)
        >>> print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Print header
    # -------------------------------------------------------------------------
    
    print("\n" + "*" * 70)
    print("*        CROSS-DATASET EVALUATION: WELFake Dataset")
    print("*" * 70)
    print(f"Dataset: {WELFAKE_DATASET_PATH}")
    print("Using load_welfake_external_eval() API from features_pipeline.py")
    
    # -------------------------------------------------------------------------
    # Step 2: Resolve configuration
    # -------------------------------------------------------------------------
    
    if limit is None:
        limit = LIMIT_SAMPLES
    
    if output_dir is None:
        output_dir = OUTPUT_TEST_DIR
    
    # -------------------------------------------------------------------------
    # Step 3: Load the model
    # -------------------------------------------------------------------------
    
    model = load_model_for_testing(model_path)
    
    # -------------------------------------------------------------------------
    # Step 4: Load and transform WELFake dataset using API
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 60)
    print("LOADING WELFAKE DATA VIA FEATURES_PIPELINE API")
    print("=" * 60)
    print(f"CSV path: {WELFAKE_DATASET_PATH}")
    print(f"Features directory: {FEATURES_DIR}")
    print(f"Using scaled features: {USE_SCALED_FEATURES}")
    
    # Use the API from features_pipeline.py
    X_wel, y_wel = load_welfake_external_eval(
        welfake_csv_path=str(WELFAKE_DATASET_PATH),
        features_out_dir=str(FEATURES_DIR),
        scaled=USE_SCALED_FEATURES,
        limit=limit
    )
    
    print(f"Total samples: {X_wel.shape[0]}")
    print(f"Feature dimensions: {X_wel.shape[1]}")
    print(f"Class distribution: fake={np.sum(y_wel == 0)}, real={np.sum(y_wel == 1)}")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 5: Run test on pre-transformed features
    # -------------------------------------------------------------------------
    
    results = run_test_on_features(
        model=model,
        X=X_wel,
        y_true=y_wel
    )
    
    # -------------------------------------------------------------------------
    # Step 6: Save results
    # -------------------------------------------------------------------------
    
    test_name = f"test_WELFake_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    saved_paths = save_test_results(
        results=results,
        titles=None,  # Not available when using pre-transformed features
        texts=None,   # Not available when using pre-transformed features
        y_true=y_wel,
        output_dir=output_dir,
        test_name=test_name,
        save_predictions=False  # Cannot save raw predictions without titles/texts
    )
    
    # -------------------------------------------------------------------------
    # Step 7: Return complete results
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 70)
    print("                    TESTING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nFinal Accuracy:       {results['metrics']['accuracy'] * 100:.2f}%")
    print(f"Final Precision:      {results['metrics']['precision'] * 100:.2f}%")
    print(f"Final Recall (Sens):  {results['metrics']['recall'] * 100:.2f}%")
    print(f"Final Specificity:    {results['metrics']['specificity'] * 100:.2f}%")
    print(f"Final ROC AUC:        {results['metrics']['auc_score']:.4f}")
    print(f"Final F1-Score:       {results['metrics']['f1_score'] * 100:.2f}%")
    print(f"Results saved to:     {output_dir}")
    print("=" * 70)
    
    complete_results = {
        'metrics': results['metrics'],
        'predictions': results['predictions'],
        'processing_time': results['processing_time'],
        'saved_paths': saved_paths
    }
    
    return complete_results


def test_single_article(
    title: str,
    text: str,
    model_path: str = None
) -> dict:
    """
    ===========================================================================
    TEST SINGLE ARTICLE
    ===========================================================================
    
    Description:
        Tests a single news article and returns the prediction.
    
    Parameters:
        title : str
            Article title.
        
        text : str
            Article body text.
        
        model_path : str, optional
            Path to the model file.
    
    Returns:
        dict : Prediction result.
    
    Example:
        >>> result = test_single_article("Breaking News", "Some text...")
        >>> print(f"Prediction: {result['label']}")
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Load model
    # -------------------------------------------------------------------------
    
    model = load_model_for_testing(model_path)
    
    # -------------------------------------------------------------------------
    # Step 2: Make prediction
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 60)
    print("TESTING SINGLE ARTICLE")
    print("=" * 60)
    print(f"Title: {title[:50]}..." if len(title) > 50 else f"Title: {title}")
    print(f"Text length: {len(text)} characters")
    
    result = predict_single(
        model=model,
        title=title,
        text=text
    )
    
    # -------------------------------------------------------------------------
    # Step 3: Print result
    # -------------------------------------------------------------------------
    
    print("\n" + "-" * 40)
    print("PREDICTION RESULT")
    print("-" * 40)
    print(f"Prediction: {result['label']}")
    
    if result['probability'] is not None:
        print(f"Probability (Fake):  {result['probability']['fake'] * 100:.2f}%")
        print(f"Probability (Real):  {result['probability']['real'] * 100:.2f}%")
    
    print("=" * 60)
    
    return result


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    """
    ===========================================================================
    MAIN EXECUTION BLOCK
    ===========================================================================
    
    This block runs when the script is executed directly from command line:
        python svm_test.py
    
    It tests the trained SVM model on the WELFake external dataset.
    
    To customize testing, either:
        1. Modify the global configuration variables at the top of this file
        2. Import this module and call test functions with custom parameters
    ===========================================================================
    """
    
    # Run test on WELFake dataset
    results = test_on_welfake()
