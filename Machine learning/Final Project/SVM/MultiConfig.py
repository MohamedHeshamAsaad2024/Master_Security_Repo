"""
================================================================================
Module Name: MultiConfig.py
================================================================================

Description:
    Multi-Configuration Training and Evaluation Script for SVM.
    
    This script automates the process of testing multiple SVM hyperparameter
    configurations. It works by:
    1. Iterating through a list of defined configurations (kernels, C, gamma, etc.)
    2. Training a model for each configuration
    3. Testing each model on the WELFake external dataset
    4. Organizing outputs into separate, timestamped folders
    5. Generating comprehensive comparison reports and plots
    
    This allows for systematic experimental comparison ("Grid Search" style)
    but with full persistence of every experiment's artifacts.

Dependencies:
    - svm_train.py: For model training logic
    - svm_test.py: For evaluation logic
    - matplotlib: For generating comparison charts

================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import sys
import time
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

# Import training and testing functions
from svm_train import train_svm_model
from svm_test import test_on_csv, find_latest_model

# Import matplotlib for comparison plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# CONFIGURATIONS TO TEST
# ==============================================================================

# List of configurations to run
# Each tuple represents a specific combination of hyperparameters:
# (config_name, kernel, C, gamma, degree, class_weight)
CONFIGS = [
    # --------------------------------------------------------------------------
    # Linear Kernel Configurations
    # --------------------------------------------------------------------------
    # Testing different regularization strengths (C parameter)
    # Effect: Higher C = less regularization (complex boundary), Lower C = more regularization (simple boundary)
    ("Linear_C1", "linear", 1.0, None, None, None),       # Standard baseline
    ("Linear_C10", "linear", 10.0, None, None, None),     # Low regularization
    ("Linear_C0.1", "linear", 0.1, None, None, None),     # High regularization
    
    # --------------------------------------------------------------------------
    # RBF Kernel Configurations
    # --------------------------------------------------------------------------
    # Testing different gamma values (influence radius of points)
    # Effect: High gamma = local influence (complex), Low gamma = global influence (smooth)
    ("RBF_scale", "rbf", 1.0, "scale", None, None),       # Auto-scaled gamma (standard)
    ("RBF_0.1", "rbf", 1.0, 0.1, None, None),             # Manual low gamma
    ("RBF_10", "rbf", 1.0, 10.0, None, None),             # Manual high gamma
    
    # --------------------------------------------------------------------------
    # Polynomial Kernel Configurations
    # --------------------------------------------------------------------------
    # Testing different polynomial degrees
    # Effect: Higher degree = more complex curves
    ("Poly_deg3", "poly", 1.0, "scale", 3, None),         # Cubic polynomial
    ("Poly_deg4", "poly", 1.0, "scale", 4, None),         # 4th degree polynomial
    
    # --------------------------------------------------------------------------
    # Sigmoid Kernel Configurations
    # --------------------------------------------------------------------------
    # Neural network like activation
    ("Sigmoid", "sigmoid", 1.0, "scale", None, None),     # Sigmoid kernel
    
    # --------------------------------------------------------------------------
    # Class Weight Configurations
    # --------------------------------------------------------------------------
    # handling imbalanced data by weighting classes
    ("Linear_balanced", "linear", 1.0, None, None, "balanced"), # Auto-balanced weights
]


# ==============================================================================
# OUTPUT DIRECTORIES
# ==============================================================================

# Timestamp for this run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Main output directory
OUTPUT_DIR = CURRENT_DIR / f"MultiConfig_Results_{TIMESTAMP}"

# The 3 folders that get created by training/testing
FOLDERS_TO_MOVE = [
    "SvmTrainedModel",
    "SvmTrainedOutput", 
    "SvmTestOutput",
]


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def convert_to_serializable(obj):
    """
    ===========================================================================
    CONVERT TO SERIALIZABLE
    ===========================================================================
    
    Description:
        Recursively converts numpy data types to standard Python types.
        Required for JSON serialization.
        
    Parameters:
        obj : any
            Object to convert (dict, list, numpy array, or primitive).
            
    Returns:
        any : JSON-serializable version of the object.
    ===========================================================================
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    else:
        return obj


def print_separator(char="=", length=60):
    """Print a separator line."""
    print(char * length)


def print_header(text):
    """Print a header with separators."""
    print_separator()
    print(text)
    print_separator()


def create_config_folder(config_name):
    """
    ===========================================================================
    CREATE CONFIG FOLDER
    ===========================================================================
    
    Description:
        Creates a dedicated folder for a specific configuration.
        
    Parameters:
        config_name : str
            Name of the configuration.
    
    Returns:
        Path : Path to the created folder.
    ===========================================================================
    """
    # Create folder path
    folder_path = OUTPUT_DIR / config_name
    
    # Create the folder
    folder_path.mkdir(parents=True, exist_ok=True)
    
    return folder_path


def clear_output_folders():
    """
    Delete the 3 output folders if they exist.
    This ensures a clean start for each configuration.
    """
    print("  Clearing previous output folders...")
    
    for folder_name in FOLDERS_TO_MOVE:
        folder_path = CURRENT_DIR / folder_name
        
        if folder_path.exists():
            try:
                shutil.rmtree(folder_path)
                print(f"    Deleted: {folder_name}/")
            except Exception as e:
                print(f"    Warning: Could not delete {folder_name}: {e}")


def move_output_folders(config_folder):
    """
    ===========================================================================
    MOVE OUTPUT FOLDERS
    ===========================================================================
    
    Description:
        Moves the standard output directories from the previous run into the
        configuration-specific folder.
        
        Uses a robust copy-then-delete strategy to handle Windows file locking
        issues that often occur with 'os.rename' or 'shutil.move'.
        
    Parameters:
        config_folder : Path
            Destination directory for this configuration.
    ===========================================================================
    """
    print("  Moving output folders...")
    
    for folder_name in FOLDERS_TO_MOVE:
        # Source folder (in SVM directory)
        source = CURRENT_DIR / folder_name
        
        # Destination folder (in config folder)
        destination = config_folder / folder_name
        
        if source.exists():
            try:
                # Copy the folder first
                if destination.exists():
                    shutil.rmtree(str(destination))
                shutil.copytree(str(source), str(destination))
                
                # Wait a moment for file handles to release
                time.sleep(0.5)
                
                # Try to delete source with retries
                for attempt in range(3):
                    try:
                        shutil.rmtree(str(source))
                        break
                    except PermissionError:
                        if attempt < 2:
                            time.sleep(1)  # Wait and retry
                        else:
                            print(f"    Warning: Could not delete source {folder_name}/ (files may be in use)")
                
                print(f"    Moved: {folder_name}/ -> {config_folder.name}/{folder_name}/")
                
            except Exception as e:
                # If copy failed but we need to proceed, just log it
                print(f"    Error moving {folder_name}/: {str(e)}")
        else:
            print(f"    Skipped: {folder_name}/ (not found)")


def save_config_info(config_folder, config_name, config_params, training_time, test_metrics, train_metrics=None):
    """
    Save configuration info and summary.
    
    Args:
        config_folder: Path to save results
        config_name: Name of the configuration
        config_params: Dictionary of configuration parameters
        training_time: Time taken for training (seconds)
        test_metrics: Metrics from testing
        train_metrics: Metrics from training (optional)
    """
    # Create results dictionary
    results_data = {
        "config_name": config_name,
        "config_params": config_params,
        "training_time_seconds": training_time,
        "test_metrics": convert_to_serializable(test_metrics),
        "train_metrics": convert_to_serializable(train_metrics) if train_metrics else {},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Save as JSON
    json_path = config_folder / "config_summary.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=4)
    
    print(f"  Summary saved to: config_summary.json")


def run_single_config(config_name, kernel, C, gamma, degree, class_weight):
    """
    Run training and testing for a single configuration.
    
    Args:
    Arguments:
        config_name : str
            Unique name for this configuration (used for folder naming).
        
        kernel : str
            SVM kernel type ('linear', 'rbf', 'poly', 'sigmoid').
            
        C : float
            Regularization parameter.
            
        gamma : str or float
            Kernel coeff. for 'rbf', 'poly', 'sigmoid'.
            
        degree : int
            Degree of the polynomial kernel function.
            
        class_weight : str or dict
            Class weights ('balanced' or dict) for imbalanced data.
        
    Returns:
        dict : Results dictionary containing:
            - config_name : str
            - success : bool
            - train_metrics : dict
            - test_metrics : dict
            - training_time : float
    ===========================================================================
    """
    print_header(f"RUNNING: {config_name}")
    
    # Print configuration
    print(f"  Kernel: {kernel}")
    print(f"  C: {C}")
    print(f"  Gamma: {gamma}")
    print(f"  Degree: {degree}")
    print(f"  Class Weight: {class_weight}")
    print()
    
    # Step 1: Clear any previous output folders
    clear_output_folders()
    
    # Step 2: Create config folder
    config_folder = create_config_folder(config_name)
    
    # Step 3: Prepare training parameters
    train_params = {
        "kernel": kernel,
        "C": C,
    }
    
    # Add optional parameters if they are set
    if gamma is not None:
        train_params["gamma"] = gamma
    
    if degree is not None:
        train_params["degree"] = degree
    
    if class_weight is not None:
        train_params["class_weight"] = class_weight
    
    # Step 4: Run training
    print()
    print("  [TRAINING]")
    start_time = time.time()
    
    try:
        train_results = train_svm_model(**train_params)
        training_time = time.time() - start_time
        
        train_metrics = train_results.get("metrics", {})
        print(f"  ✓ Training completed in {training_time:.2f} seconds")
        print(f"    Train Accuracy: {train_metrics.get('accuracy', 0) * 100:.2f}%")
        
    except Exception as e:
        print(f"  ✗ Training failed: {str(e)}")
        training_time = time.time() - start_time
        train_metrics = {}
    
    # Step 5: Run testing on WELFake dataset
    print()
    print("  [TESTING on WELFake]")
    
    test_metrics = {}
    try:
        # Find the model we just trained
        model_path = find_latest_model()
        
        if model_path:
            # Path to WELFake dataset
            welfake_path = CURRENT_DIR.parent / "Data_preprocessing_and_cleanup" / "External_Datasets" / "WELFake_Dataset.csv"
            
            if welfake_path.exists():
                # Run test (CSV must have columns: title, text, label)
                test_results = test_on_csv(
                    csv_path=str(welfake_path),
                    model_path=str(model_path)
                )
                
                test_metrics = test_results.get("metrics", {})
                print(f"  ✓ Testing completed")
                print(f"    Test Accuracy: {test_metrics.get('accuracy', 0) * 100:.2f}%")
            else:
                print(f"  ⚠ WELFake dataset not found at: {welfake_path}")
        else:
            print("  ⚠ No model found to test")
            
    except Exception as e:
        print(f"  ✗ Testing failed: {str(e)}")
    
    # Step 6: Move output folders to config folder
    print()
    move_output_folders(config_folder)
    
    # Step 7: Save config summary
    save_config_info(
        config_folder=config_folder,
        config_name=config_name,
        config_params=train_params,
        training_time=training_time,
        test_metrics=test_metrics,
        train_metrics=train_metrics
    )
    
    print()
    
    # Return results for comparison
    return {
        "config_name": config_name,
        "success": len(train_metrics) > 0,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "training_time": training_time,
    }


def create_comparison_plot(all_results):
    """
    Create a bar chart comparing all configurations.
    """
    print_header("CREATING COMPARISON PLOT")
    
    # Filter successful results
    successful = [r for r in all_results if r.get("success", False)]
    
    if not successful:
        print("  No successful results to plot")
        return
    
    # Use output dir from environment or assume global OUTPUT_DIR
    # (In dynamic load case, OUTPUT_DIR might need to be adjusted, but usually global is fine 
    # if we set it, or we use current working dir relative)
    # Ideally, we save plot to the same dir where results came from.
    # For now, we assume global OUTPUT_DIR is set correctly before calling this.
    
    try:
        # Extract data for plotting
        config_names = [r["config_name"] for r in successful]
        
        # Training metrics
        train_acc = [r["train_metrics"].get("accuracy", 0) * 100 for r in successful]
        
        # Test metrics
        test_acc = [r["test_metrics"].get("accuracy", 0) * 100 for r in successful]
        test_prec = [r["test_metrics"].get("precision", 0) * 100 for r in successful]
        test_recall = [r["test_metrics"].get("recall", 0) * 100 for r in successful]
        test_f1 = [r["test_metrics"].get("f1_score", 0) * 100 for r in successful]
        
        # Create figure with three subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Subplot 1: Train vs Test Accuracy
        x = np.arange(len(config_names))
        width = 0.35
        
        ax1 = axes[0]
        ax1.bar(x - width/2, train_acc, width, label='Train Accuracy', color='#2196F3')
        ax1.bar(x + width/2, test_acc, width, label='Test Accuracy', color='#4CAF50')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Train vs Test Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(config_names, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 105)
        ax1.grid(axis='y', alpha=0.3)
        
        # Subplot 2: All Test Metrics
        width = 0.15
        ax2 = axes[1]
        ax2.bar(x - 2.5*width, test_acc, width, label='Accuracy', color='#2196F3')
        ax2.bar(x - 1.5*width, test_prec, width, label='Precision', color='#4CAF50')
        ax2.bar(x - 0.5*width, test_recall, width, label='Recall (Sens)', color='#FF9800')
        ax2.bar(x + 0.5*width, [r["test_metrics"].get("specificity", 0) * 100 for r in successful], width, label='Specificity', color='#009688')
        ax2.bar(x + 1.5*width, [r["test_metrics"].get("auc_score", 0) * 100 for r in successful], width, label='AUC', color='#673AB7')
        ax2.bar(x + 2.5*width, test_f1, width, label='F1-Score', color='#9C27B0')
        ax2.set_ylabel('Score (%)')
        ax2.set_title('Test Metrics Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(config_names, rotation=45, ha='right')
        ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))
        ax2.set_ylim(0, 105)
        ax2.grid(axis='y', alpha=0.3)
        
        # Subplot 3: Training Time
        training_times = [r["training_time"] for r in successful]
        ax3 = axes[2]
        bars = ax3.bar(config_names, training_times, color='#E91E63')
        ax3.set_ylabel('Training Time (seconds)')
        ax3.set_title('Training Time Comparison')
        ax3.set_xticks(x)   # Explicitly set ticks equal to x positions
        ax3.set_xticklabels(config_names, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, time_val in zip(bars, training_times):
            ax3.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f'{time_val:.1f}s',
                ha='center',
                va='bottom',
                fontsize=8
            )
        
        plt.tight_layout()
        
        plot_path = OUTPUT_DIR / "comparison_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Comparison plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"  Warning: Could not create comparison plot: {e}")


def create_comparison_report(all_results):
    """
    Create a text report comparing all configurations.
    """
    print_header("CREATING COMPARISON REPORT")
    
    # Create report content
    lines = []
    lines.append("=" * 100)
    lines.append("MULTI-CONFIGURATION SVM COMPARISON REPORT")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Configurations: {len(all_results)}")
    lines.append("")
    
    # Summary table header
    lines.append("-" * 100)
    lines.append(f"{'Config Name':<18} {'Train Acc':>10} {'Test Acc':>10} {'Precision':>10} {'Rec(Sens)':>10} {'Spec':>10} {'AUC':>10} {'F1-Score':>10} {'Time(s)':>10}")
    lines.append("-" * 125)
    
    # Add each result
    for result in all_results:
        if result.get("success", False):
            train_m = result.get("train_metrics", {})
            test_m = result.get("test_metrics", {})
            lines.append(
                f"{result['config_name']:<18} "
                f"{train_m.get('accuracy', 0)*100:>9.2f}% "
                f"{test_m.get('accuracy', 0)*100:>9.2f}% "
                f"{test_m.get('precision', 0)*100:>9.2f}% "
                f"{test_m.get('recall', 0)*100:>9.2f}% "
                f"{test_m.get('specificity', 0)*100:>9.2f}% "
                f"{test_m.get('auc_score', 0):>10.4f} "
                f"{test_m.get('f1_score', 0)*100:>9.2f}% "
                f"{result.get('training_time', 0):>10.1f}"
            )
        else:
            lines.append(f"{result['config_name']:<18} FAILED")
    
    lines.append("-" * 125)
    lines.append("")
    
    # Find best configurations
    successful = [r for r in all_results if r.get("success", False)]
    if successful:
        lines.append("BEST CONFIGURATIONS:")
        lines.append("-" * 50)
        
        # Best by test accuracy
        best_test = max(successful, key=lambda x: x.get("test_metrics", {}).get("accuracy", 0))
        lines.append(f"  Best Test Accuracy:  {best_test['config_name']} ({best_test.get('test_metrics', {}).get('accuracy', 0)*100:.2f}%)")
        
        # Best by F1
        best_f1 = max(successful, key=lambda x: x.get("test_metrics", {}).get("f1_score", 0))
        lines.append(f"  Best F1-Score:       {best_f1['config_name']} ({best_f1.get('test_metrics', {}).get('f1_score', 0)*100:.2f}%)")
        
        # Best AUC
        best_auc = max(successful, key=lambda x: x.get("test_metrics", {}).get("auc_score", 0))
        lines.append(f"  Best AUC Score:      {best_auc['config_name']} ({best_auc.get('test_metrics', {}).get('auc_score', 0):.4f})")
        
        # Fastest
        fastest = min(successful, key=lambda x: x.get("training_time", float('inf')))
        lines.append(f"  Fastest Training:    {fastest['config_name']} ({fastest.get('training_time', 0):.1f} seconds)")
    
    lines.append("")
    lines.append("=" * 100)
    
    # Write report
    report_content = "\n".join(lines)
    report_path = OUTPUT_DIR / "comparison_report.txt"
    
    with open(report_path, "w") as f:
        f.write(report_content)
    
    # Also print to console
    print(report_content)
    print(f"\n  Report saved to: {report_path}")
    
    # Save as JSON too - using global convert_to_serializable
    json_path = OUTPUT_DIR / "all_results.json"
    try:
        with open(json_path, "w") as f:
            json.dump(convert_to_serializable(all_results), f, indent=4)
        print(f"  JSON data saved to: {json_path}")
    except Exception as e:
        print(f"  Warning: Could not save all_results.json: {e}")


def reconstruct_results_from_folder(target_dir):
    """
    Reconstruct 'all_results' list from an existing output directory.
    Scans subfolders for config_summary.json and other metadata.
    """
    path = Path(target_dir)
    print_header(f"RECONSTRUCTING RESULTS FROM: {path}")
    
    if not path.exists():
        print(f"Error: Directory not found: {path}")
        return []
    
    all_results = []
    
    # Iterate through all subdirectories
    for item in path.iterdir():
        if item.is_dir():
            config_name = item.name
            summary_path = item / "config_summary.json"
            
            if summary_path.exists():
                try:
                    with open(summary_path, "r") as f:
                        data = json.load(f)
                    
                    # Basic info
                    training_time = data.get("training_time_seconds", 0)
                    test_metrics = data.get("test_metrics", {})
                    train_metrics = data.get("train_metrics", {})
                    
                    # If train_metrics is empty in summary (old format), try to find it in SvmTrainedModel
                    if not train_metrics:
                        model_dir = item / "SvmTrainedModel"
                        if model_dir.exists():
                            # Find metadata json
                            meta_files = list(model_dir.glob("*_metadata.json"))
                            if meta_files:
                                try:
                                    with open(meta_files[0], "r") as f_meta:
                                        meta_data = json.load(f_meta)
                                        if "metrics" in meta_data:
                                            train_metrics = meta_data["metrics"]
                                        else:
                                            # Flat structure (metrics at top level)
                                            train_metrics = meta_data
                                except:
                                    pass
                    
                    result = {
                        "config_name": config_name,
                        "success": True,
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                        "training_time": training_time
                    }
                    all_results.append(result)
                    print(f"  Loaded: {config_name}")
                    
                except Exception as e:
                    print(f"  Error loading {config_name}: {e}")
            else:
                # Not a config folder, skip
                pass
                
    return all_results


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Main function to run all configurations."""
    
    # Check if user passed an argument to run comparison only
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        
        # If it's a directory, run comparison only
        if os.path.isdir(target_dir):
            global OUTPUT_DIR
            OUTPUT_DIR = Path(target_dir)
            
            all_results = reconstruct_results_from_folder(target_dir)
            
            if all_results:
                create_comparison_plot(all_results)
                create_comparison_report(all_results)
                print_header("RECOVERY COMPLETE")
            else:
                print("No valid results found in directory.")
            return

    print_header("MULTI-CONFIGURATION SVM TRAINING")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Number of Configurations: {len(CONFIGS)}")
    print()
    print("Each configuration will:")
    print("  1. Clear previous output folders")
    print("  2. Train the SVM model")
    print("  3. Test on WELFake dataset")
    print("  4. Move all outputs to config folder")
    print()
    print("NOTE: To regenerate report from existing folder, run:")
    print(f"python MultiConfig.py <path_to_results_folder>")
    print()
    
    # Create main output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = []
    
    # Record total start time
    total_start = time.time()
    
    # Run each configuration
    for i, config in enumerate(CONFIGS):
        # Unpack configuration
        config_name, kernel, C, gamma, degree, class_weight = config
        
        print(f"\n[{i+1}/{len(CONFIGS)}] Starting configuration: {config_name}")
        
        # Run this configuration
        result = run_single_config(
            config_name=config_name,
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            class_weight=class_weight
        )
        
        # Store result
        all_results.append(result)
    
    # Calculate total time
    total_time = time.time() - total_start
    
    print_header("ALL CONFIGURATIONS COMPLETED")
    print(f"Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print()
    
    # Create comparison outputs
    create_comparison_plot(all_results)
    create_comparison_report(all_results)
    
    print_header("DONE")
    print(f"All results saved to: {OUTPUT_DIR}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
