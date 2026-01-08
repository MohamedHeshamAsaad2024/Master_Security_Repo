"""
================================================================================
Module Name: svm_train.py
================================================================================

Description:
    Support Vector Machine (SVM) Training Module for Fake News Classification.
    
    This module provides a complete SVM implementation with:
    - Configurable kernel types (Linear, RBF, Polynomial, Sigmoid)
    - Full documentation of all hyperparameters with mathematical equations
    - API functions for easy integration
    - Command-line interface for standalone usage
    
Dependencies:
    - features_pipeline.py (for loading preprocessed features)
    - scikit-learn (for SVM implementation)
    - joblib (for model persistence)
    - matplotlib (for visualization)

================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# Standard library imports
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Third-party imports
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
from joblib import dump, load

# Add parent directory to path to import features_pipeline
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_PREPROCESSING_DIR = PROJECT_ROOT / "Data_preprocessing_and_cleanup"
sys.path.insert(0, str(DATA_PREPROCESSING_DIR))

from features_pipeline import (
    load_feature_matrices,
    load_artifacts,
    transform_records,
    FeatureConfig
)


# ==============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ==============================================================================
# 
# This section contains all configurable parameters for the SVM model.
# Each configuration includes:
#   - DESCRIPTION: What this configuration controls
#   - EQUATION: Mathematical formula used (if applicable)
#   - AFFECTED STEPS: Which parts of the pipeline use this configuration
#   - CALLED BY: Which function(s) use this configuration
#   - POSSIBLE VALUES: Options you can test
#   - EXPECTED BEHAVIOR: What happens with different values
#   - DEPENDENCIES: Other configurations that affect or are affected by this one
# ==============================================================================


# ------------------------------------------------------------------------------
# OUTPUT PATH CONFIGURATIONS
# ------------------------------------------------------------------------------

# CONFIGURATION: OUTPUT_MODEL_DIR
# DESCRIPTION: Directory path where trained SVM models will be saved.
#              Models are saved as .joblib files for later loading and prediction.
# AFFECTED STEPS: Save Model step
# CALLED BY: save_model(), train_svm_model()
# POSSIBLE VALUES: Any valid directory path (relative or absolute)
# EXPECTED BEHAVIOR: 
#   - Directory will be created if it doesn't exist
#   - Model files will be saved with timestamp in filename
# DEPENDENCIES: None
OUTPUT_MODEL_DIR = CURRENT_DIR / "SvmTrainedModel"


# CONFIGURATION: OUTPUT_PLOTS_DIR
# DESCRIPTION: Directory path where training plots and visualizations will be saved.
#              Includes confusion matrix and other performance plots.
# AFFECTED STEPS: Evaluation step, Visualization step
# CALLED BY: plot_confusion_matrix(), train_svm_model()
# POSSIBLE VALUES: Any valid directory path (relative or absolute)
# EXPECTED BEHAVIOR:
#   - Directory will be created if it doesn't exist
#   - Plots saved as PNG files with descriptive names
# DEPENDENCIES: None
OUTPUT_PLOTS_DIR = CURRENT_DIR / "SvmTrainedOutput"


# CONFIGURATION: FEATURES_DIR
# DESCRIPTION: Directory path where preprocessed feature matrices are stored.
#              This is the output directory from features_pipeline.py
# AFFECTED STEPS: Load Data step
# CALLED BY: load_training_data(), train_svm_model()
# POSSIBLE VALUES: Path to directory containing X_train_scaled.npz, y_train.csv, etc.
# EXPECTED BEHAVIOR:
#   - Must contain valid feature matrices from features_pipeline.py
#   - Will raise error if files are missing
# DEPENDENCIES: Requires running features_pipeline.py first
FEATURES_DIR = DATA_PREPROCESSING_DIR / "Output" / "features_out"


# ------------------------------------------------------------------------------
# SVM KERNEL CONFIGURATION
# ------------------------------------------------------------------------------

# CONFIGURATION: KERNEL_TYPE
# DESCRIPTION: The kernel function used for SVM classification.
#              Kernels transform data into higher-dimensional space for better separation.
#
# EQUATION (for each kernel):
#   - "linear":  K(x, x') = x · x'  (dot product)
#   - "rbf":     K(x, x') = exp(-gamma * ||x - x'||²)  (Radial Basis Function)
#   - "poly":    K(x, x') = (gamma * x · x' + coef0)^degree  (Polynomial)
#   - "sigmoid": K(x, x') = tanh(gamma * x · x' + coef0)  (Sigmoid/tanh)
#
# AFFECTED STEPS: Training step, Prediction step
# CALLED BY: create_svm_model(), train_svm_model()
#
# POSSIBLE VALUES AND EXPECTED BEHAVIOR:
#   - "linear": 
#       * Best for: Linearly separable data or high-dimensional sparse data (like text)
#       * Pros: Fast training, interpretable, works well with TF-IDF features
#       * Cons: Cannot capture non-linear patterns
#       * Expected: Usually 90-95% accuracy on fake news task
#   - "rbf" (Radial Basis Function):
#       * Best for: Non-linear decision boundaries, complex patterns
#       * Pros: Very flexible, often achieves highest accuracy
#       * Cons: Slower training, prone to overfitting, harder to interpret
#       * Expected: Can reach 95-97% accuracy but slower training
#   - "poly" (Polynomial):
#       * Best for: Data with polynomial relationships
#       * Pros: More flexible than linear, controllable complexity via degree
#       * Cons: Can be unstable with high degree, slower than linear
#       * Expected: 92-95% accuracy depending on degree
#   - "sigmoid":
#       * Best for: Behaves like neural network activation
#       * Pros: Similar to two-layer perceptron
#       * Cons: Not always positive semi-definite, can be unstable
#       * Expected: Usually 88-93% accuracy, less stable than others
#
# DEPENDENCIES: 
#   - If "rbf" or "poly" or "sigmoid": GAMMA parameter is used
#   - If "poly": DEGREE and COEF0 parameters are used
#   - If "sigmoid": COEF0 parameter is used
KERNEL_TYPE = "linear"


# CONFIGURATION: C_REGULARIZATION
# DESCRIPTION: Regularization parameter that controls the trade-off between 
#              achieving a low training error and a low testing error (generalization).
#
# EQUATION: 
#   SVM Optimization: minimize (1/2)||w||² + C * Σ ξᵢ
#   Where:
#     - ||w||² is the margin size (larger = simpler model)
#     - ξᵢ are slack variables (allow misclassifications)
#     - C weights how much we penalize misclassifications
#
# AFFECTED STEPS: Training step
# CALLED BY: create_svm_model(), train_svm_model()
#
# POSSIBLE VALUES AND EXPECTED BEHAVIOR:
#   - 0.01: Very high regularization (very simple model)
#       * Underfitting likely, low training accuracy, may generalize better
#   - 0.1: High regularization
#       * More tolerance for errors, simpler decision boundary
#   - 1.0: Default, balanced (RECOMMENDED starting point)
#       * Good balance between fitting and regularization
#   - 10.0: Low regularization
#       * Less tolerance for training errors, more complex boundary
#   - 100.0: Very low regularization
#       * Overfitting likely, high training accuracy, poor generalization
#
# DEPENDENCIES: None (affects all kernel types)
C_REGULARIZATION = 1.0


# CONFIGURATION: GAMMA
# DESCRIPTION: Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels.
#              Controls how far the influence of a single training example reaches.
#
# EQUATION:
#   - RBF: K(x, x') = exp(-gamma * ||x - x'||²)
#   - Gamma defines the "reach" of each training point
#
# AFFECTED STEPS: Training step (only when KERNEL_TYPE is not "linear")
# CALLED BY: create_svm_model(), train_svm_model()
#
# POSSIBLE VALUES AND EXPECTED BEHAVIOR:
#   - "scale": gamma = 1 / (n_features * X.var())  (RECOMMENDED)
#       * Automatically scales based on data variance
#       * Works well in most cases without tuning
#   - "auto": gamma = 1 / n_features
#       * Simpler scaling, ignores data variance
#   - 0.001: Very low gamma (large influence radius)
#       * Smoother decision boundary, may underfit
#   - 0.01: Low gamma
#       * More tolerance, simpler boundaries
#   - 0.1: Medium gamma
#       * Moderate complexity
#   - 1.0: High gamma (small influence radius)
#       * Complex decision boundary, may overfit
#   - 10.0: Very high gamma
#       * Each point has very local influence, severe overfitting
#
# DEPENDENCIES: Only used when KERNEL_TYPE is "rbf", "poly", or "sigmoid"
GAMMA = "scale"


# CONFIGURATION: DEGREE
# DESCRIPTION: Degree of the polynomial kernel function.
#              Only used when KERNEL_TYPE = "poly".
#
# EQUATION:
#   Polynomial Kernel: K(x, x') = (gamma * x · x' + coef0)^degree
#
# AFFECTED STEPS: Training step (only when KERNEL_TYPE = "poly")
# CALLED BY: create_svm_model(), train_svm_model()
#
# POSSIBLE VALUES AND EXPECTED BEHAVIOR:
#   - 2: Quadratic polynomial
#       * Models quadratic relationships
#       * Fast training, good for moderately complex data
#   - 3: Cubic polynomial (RECOMMENDED for poly kernel)
#       * More complex patterns, standard default
#       * Good balance of complexity and speed
#   - 4: Fourth-degree polynomial
#       * More complex decision boundaries
#       * Slower training, higher risk of overfitting
#   - 5+: Higher degree polynomials
#       * Very complex boundaries, usually overfits
#       * Slow training, numerically unstable
#
# DEPENDENCIES: Only used when KERNEL_TYPE = "poly"
DEGREE = 3


# CONFIGURATION: COEF0
# DESCRIPTION: Independent term in kernel function.
#              Used in "poly" and "sigmoid" kernels.
#
# EQUATION:
#   - Poly: K(x, x') = (gamma * x · x' + coef0)^degree
#   - Sigmoid: K(x, x') = tanh(gamma * x · x' + coef0)
#
# AFFECTED STEPS: Training step (only when KERNEL_TYPE = "poly" or "sigmoid")
# CALLED BY: create_svm_model(), train_svm_model()
#
# POSSIBLE VALUES AND EXPECTED BEHAVIOR:
#   - 0.0: No independent term (standard)
#       * Kernel passes through origin
#   - 1.0: Adds constant to kernel
#       * Shifts decision boundary
#   - Higher values: Further shifts kernel behavior
#
# DEPENDENCIES: Only used when KERNEL_TYPE = "poly" or "sigmoid"
COEF0 = 0.0


# ------------------------------------------------------------------------------
# TRAINING CONFIGURATION
# ------------------------------------------------------------------------------

# CONFIGURATION: USE_SCALED_FEATURES
# DESCRIPTION: Whether to use scaled (StandardScaler) or unscaled TF-IDF features.
#              Scaling normalizes features to have zero mean and unit variance.
#
# EQUATION:
#   StandardScaler: z = (x - μ) / σ
#   Where:
#     - x is the original feature value
#     - μ is the mean of the feature
#     - σ is the standard deviation of the feature
#
# AFFECTED STEPS: Load Data step
# CALLED BY: load_training_data(), train_svm_model()
#
# POSSIBLE VALUES AND EXPECTED BEHAVIOR:
#   - True (RECOMMENDED for SVM):
#       * Features are normalized to similar scales
#       * SVM converges faster and performs better
#       * Required for SVM when features have different scales
#   - False:
#       * Raw TF-IDF values are used
#       * May cause SVM to favor features with larger values
#       * Only use if features are already normalized
#
# DEPENDENCIES: Requires scaler.joblib artifact from features_pipeline.py
USE_SCALED_FEATURES = True


# CONFIGURATION: ENABLE_PROBABILITY
# DESCRIPTION: Whether to enable probability estimates for predictions.
#              When enabled, model can output confidence scores for predictions.
#
# EQUATION:
#   Uses Platt scaling (sigmoid calibration):
#   P(y=1|f) = 1 / (1 + exp(A*f + B))
#   Where f is the SVM decision function output, A and B are fitted parameters.
#
# AFFECTED STEPS: Training step, Evaluation step
# CALLED BY: create_svm_model()
#
# POSSIBLE VALUES AND EXPECTED BEHAVIOR:
#   - True:
#       * Enables predict_proba() and decision function calibration
#       * Provides probability scores for predictions
#       * Slows down training significantly (internal cross-validation)
#   - False:
#       * Faster training
#       * Cannot compute probability outputs
#
# DEPENDENCIES: None
ENABLE_PROBABILITY = False


# CONFIGURATION: CLASS_WEIGHT
# DESCRIPTION: Weights associated with classes to handle imbalanced datasets.
#              Adjusts the C parameter for each class.
#
# EQUATION:
#   Weighted SVM: minimize (1/2)||w||² + C * Σ wᵢ * ξᵢ
#   Where wᵢ is the class weight for sample i.
#   With "balanced": wⱼ = n_samples / (n_classes * n_samples_j)
#
# AFFECTED STEPS: Training step
# CALLED BY: create_svm_model(), train_svm_model()
#
# POSSIBLE VALUES AND EXPECTED BEHAVIOR:
#   - None:
#       * All classes have equal weight
#       * May be biased toward majority class if data is imbalanced
#   - "balanced":
#       * Automatically adjusts weights inversely proportional to class frequencies
#       * Helps with imbalanced datasets
#       * May reduce accuracy on majority class
#   - {0: 1.0, 1: 2.0}: Custom weights
#       * Manually specify importance of each class
#       * Higher weight = more penalty for misclassifying that class
#
# DEPENDENCIES: None
CLASS_WEIGHT = None


# CONFIGURATION: RANDOM_STATE
# DESCRIPTION: Random seed for reproducibility.
#              Controls randomness in training process.
#
# AFFECTED STEPS: Training step
# CALLED BY: create_svm_model(), train_svm_model()
#
# POSSIBLE VALUES:
#   - Any integer: Fixed seed for reproducible results
#   - None: Random behavior each run
#
# EXPECTED BEHAVIOR:
#   - Same seed = same model (given same data)
#   - Different seeds may give slightly different results
#
# DEPENDENCIES: None
RANDOM_STATE = 42


# CONFIGURATION: MAX_ITERATIONS
# DESCRIPTION: Maximum number of iterations for the solver.
#              Limits training time for large datasets.
#
# AFFECTED STEPS: Training step
# CALLED BY: create_svm_model(), train_svm_model()
#
# POSSIBLE VALUES AND EXPECTED BEHAVIOR:
#   - -1: No limit (train until convergence)
#       * Best for final training
#       * May take very long for large datasets
#   - 1000: Short training limit
#       * May not converge, lower accuracy
#   - 10000: Medium limit
#       * Usually sufficient for most datasets
#   - 100000: Long training limit
#       * Almost always converges
#
# DEPENDENCIES: None
MAX_ITERATIONS = -1


# CONFIGURATION: VERBOSE
# DESCRIPTION: Enable verbose output during training.
#              Shows progress and convergence information.
#
# AFFECTED STEPS: Training step
# CALLED BY: create_svm_model()
#
# POSSIBLE VALUES:
#   - True: Show detailed training progress
#   - False: Silent training
#
# DEPENDENCIES: None
VERBOSE = False


# ==============================================================================
# API FUNCTIONS
# ==============================================================================
# 
# These functions provide a clean interface for external code (GUI, scripts, etc.)
# to use the SVM model. Each function is self-contained and documented.
# ==============================================================================


def load_training_data(
    features_dir: str = None,
    use_scaled: bool = None
) -> tuple:
    """
    ===========================================================================
    LOAD TRAINING DATA
    ===========================================================================
    
    Description:
        Loads the preprocessed feature matrices and labels from the features
        pipeline output directory.
    
    Parameters:
        features_dir : str, optional
            Path to the features output directory.
            If None, uses the global FEATURES_DIR configuration.
        
        use_scaled : bool, optional
            Whether to load scaled or unscaled features.
            If None, uses the global USE_SCALED_FEATURES configuration.
    
    Returns:
        tuple : (X_train, X_test, y_train, y_test)
            - X_train : scipy.sparse.csr_matrix
                Training feature matrix (n_train_samples, n_features)
            - X_test : scipy.sparse.csr_matrix
                Test feature matrix (n_test_samples, n_features)
            - y_train : numpy.ndarray
                Training labels (0=fake, 1=real)
            - y_test : numpy.ndarray
                Test labels (0=fake, 1=real)
    
    Raises:
        FileNotFoundError: If feature files are not found in the directory.
    
    Example:
        >>> X_train, X_test, y_train, y_test = load_training_data()
        >>> print(f"Training samples: {X_train.shape[0]}")
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Resolve configuration values
    # -------------------------------------------------------------------------
    
    # If features_dir is not provided, use the global configuration
    if features_dir is None:
        features_dir = FEATURES_DIR
    
    # If use_scaled is not provided, use the global configuration
    if use_scaled is None:
        use_scaled = USE_SCALED_FEATURES
    
    # -------------------------------------------------------------------------
    # Step 2: Convert to Path object for safe path handling
    # -------------------------------------------------------------------------
    
    # Convert string to Path object if necessary
    features_path = Path(features_dir)
    
    # -------------------------------------------------------------------------
    # Step 3: Validate that the directory exists
    # -------------------------------------------------------------------------
    
    # Check if the features directory exists
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features directory not found: {features_path}\n"
            f"Please run the features_pipeline.py first to generate features."
        )
    
    # -------------------------------------------------------------------------
    # Step 4: Load feature matrices and labels
    # -------------------------------------------------------------------------
    
    # Print loading status
    print("=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)
    print(f"Features directory: {features_path}")
    print(f"Using scaled features: {use_scaled}")
    
    # Load the matrices using the features_pipeline function
    # This function loads:
    #   - X_train: Training feature matrix (sparse CSR format)
    #   - X_test: Test feature matrix (sparse CSR format)
    #   - y_train: Training labels (numpy array)
    #   - y_test: Test labels (numpy array)
    X_train, X_test, y_train, y_test = load_feature_matrices(
        out_dir=features_path,
        scaled=use_scaled
    )
    
    # -------------------------------------------------------------------------
    # Step 5: Print data summary
    # -------------------------------------------------------------------------
    
    print(f"\nData loaded successfully!")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Training class distribution: fake={np.sum(y_train == 0)}, real={np.sum(y_train == 1)}")
    print(f"Test class distribution: fake={np.sum(y_test == 0)}, real={np.sum(y_test == 1)}")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 6: Return the loaded data
    # -------------------------------------------------------------------------
    
    return X_train, X_test, y_train, y_test


def create_svm_model(
    kernel: str = None,
    C: float = None,
    gamma: str = None,
    degree: int = None,
    coef0: float = None,
    class_weight: str = None,
    probability: bool = None,
    random_state: int = None,
    max_iter: int = None,
    verbose: bool = None
) -> SVC:
    """
    ===========================================================================
    CREATE SVM MODEL
    ===========================================================================
    
    Description:
        Creates an SVM classifier with the specified hyperparameters.
        Uses global configuration values if parameters are not provided.
    
    Parameters:
        kernel : str, optional
            Kernel type: 'linear', 'rbf', 'poly', or 'sigmoid'.
            Default: KERNEL_TYPE global configuration.
        
        C : float, optional
            Regularization parameter.
            Default: C_REGULARIZATION global configuration.
        
        gamma : str or float, optional
            Kernel coefficient for 'rbf', 'poly', 'sigmoid'.
            Default: GAMMA global configuration.
        
        degree : int, optional
            Degree for polynomial kernel.
            Default: DEGREE global configuration.
        
        coef0 : float, optional
            Independent term in kernel function.
            Default: COEF0 global configuration.
        
        class_weight : str or dict, optional
            Class weights for handling imbalanced data.
            Default: CLASS_WEIGHT global configuration.
        
        probability : bool, optional
            Enable probability estimates.
            Default: ENABLE_PROBABILITY global configuration.
        
        random_state : int, optional
            Random seed for reproducibility.
            Default: RANDOM_STATE global configuration.
        
        max_iter : int, optional
            Maximum iterations for solver.
            Default: MAX_ITERATIONS global configuration.
        
        verbose : bool, optional
            Enable verbose output.
            Default: VERBOSE global configuration.
    
    Returns:
        SVC : sklearn.svm.SVC
            Configured SVM classifier (not yet fitted).
    
    Example:
        >>> model = create_svm_model(kernel='rbf', C=10.0)
        >>> print(model.get_params())
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Resolve configuration values (use globals if not provided)
    # -------------------------------------------------------------------------
    
    # Kernel type configuration
    if kernel is None:
        kernel = KERNEL_TYPE
    
    # Regularization parameter
    if C is None:
        C = C_REGULARIZATION
    
    # Gamma parameter (kernel coefficient)
    if gamma is None:
        gamma = GAMMA
    
    # Polynomial degree
    if degree is None:
        degree = DEGREE
    
    # Coef0 parameter
    if coef0 is None:
        coef0 = COEF0
    
    # Class weight configuration
    if class_weight is None:
        class_weight = CLASS_WEIGHT
    
    # Probability estimation
    if probability is None:
        probability = ENABLE_PROBABILITY
    
    # Random state for reproducibility
    if random_state is None:
        random_state = RANDOM_STATE
    
    # Maximum iterations
    if max_iter is None:
        max_iter = MAX_ITERATIONS
    
    # Verbose output
    if verbose is None:
        verbose = VERBOSE
    
    # -------------------------------------------------------------------------
    # Step 2: Print model configuration
    # -------------------------------------------------------------------------
    
    print("=" * 60)
    print("CREATING SVM MODEL")
    print("=" * 60)
    
    print(f"Kernel: {kernel}")
    print(f"C (Regularization): {C}")
    
    # Only print gamma if applicable
    if kernel in ['rbf', 'poly', 'sigmoid']:
        print(f"Gamma: {gamma}")
    
    # Only print degree if polynomial kernel
    if kernel == 'poly':
        print(f"Degree: {degree}")
    
    # Only print coef0 if applicable
    if kernel in ['poly', 'sigmoid']:
        print(f"Coef0: {coef0}")
    
    print(f"Class Weight: {class_weight}")
    print(f"Probability: {probability}")
    print(f"Random State: {random_state}")
    print(f"Max Iterations: {max_iter}")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 3: Create SVM classifier
    # -------------------------------------------------------------------------
    
    # sklearn.svm.SVC implements Support Vector Classification
    # 
    # The SVM optimization problem:
    #   minimize (1/2)||w||² + C * Σ ξᵢ
    #   subject to: yᵢ(w·xᵢ + b) >= 1 - ξᵢ
    #               ξᵢ >= 0
    # 
    # Where:
    #   - w is the weight vector (defines the hyperplane)
    #   - b is the bias term
    #   - ξᵢ are slack variables (allow soft margin)
    #   - C controls the trade-off between margin and errors
    
    svm_model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        class_weight=class_weight,
        probability=probability,
        random_state=random_state,
        max_iter=max_iter,
        verbose=verbose
    )
    # -------------------------------------------------------------------------
    # Step 4: Return the configured model
    # -------------------------------------------------------------------------
    
    print("\nSVM model created successfully!")
    
    return svm_model


def train_model(
    model: SVC,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> SVC:
    """
    ===========================================================================
    TRAIN MODEL
    ===========================================================================
    
    Description:
        Fits the SVM model to the training data.
    
    Parameters:
        model : SVC
            The SVM classifier to train (created by create_svm_model).
        
        X_train : numpy.ndarray or scipy.sparse matrix
            Training feature matrix.
        
        y_train : numpy.ndarray
            Training labels.
    
    Returns:
        SVC : The trained SVM model.
    
    Example:
        >>> model = create_svm_model()
        >>> model = train_model(model, X_train, y_train)
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Print training start message
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 60)
    print("TRAINING SVM MODEL")
    print("=" * 60)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print("\nTraining in progress...")
    
    # -------------------------------------------------------------------------
    # Step 2: Record training start time
    # -------------------------------------------------------------------------
    
    import time
    start_time = time.time()
    
    # -------------------------------------------------------------------------
    # Step 3: Fit the model to training data
    # -------------------------------------------------------------------------
    
    # The fit() method learns the decision boundary from training data
    # 
    # For linear SVM, it finds the hyperplane:
    #   f(x) = w · x + b
    # 
    # For non-linear SVM (with kernel K), it finds:
    #   f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b
    # 
    # Where:
    #   - αᵢ are the learned dual coefficients
    #   - K(xᵢ, x) is the kernel function
    #   - Only support vectors (points with αᵢ > 0) contribute
    
    model.fit(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # Step 4: Record training time
    # -------------------------------------------------------------------------
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # -------------------------------------------------------------------------
    # Step 5: Print training summary
    # -------------------------------------------------------------------------
    
    print(f"\nTraining completed!")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Number of support vectors: {model.n_support_.sum()}")
    print(f"Support vectors per class: {dict(zip(['fake', 'real'], model.n_support_))}")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 6: Return the trained model
    # -------------------------------------------------------------------------
    
    return model


def evaluate_model(
    model: SVC,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    ===========================================================================
    EVALUATE MODEL
    ===========================================================================
    
    Description:
        Evaluates the trained SVM model on test data and returns all metrics.
    
    Parameters:
        model : SVC
            Trained SVM classifier.
        
        X_test : numpy.ndarray or scipy.sparse matrix
            Test feature matrix.
        
        y_test : numpy.ndarray
            True test labels.
    
    Returns:
        dict : Dictionary containing all evaluation metrics:
            - accuracy : float
            - precision : float
            - recall : float (Sensitivity)
            - specificity : float
            - f1_score : float
            - auc_score : float
            - confusion_matrix : numpy.ndarray
            - classification_report : str
            - predictions : numpy.ndarray
            - y_score : numpy.ndarray (for ROC/AUC)
    
    Equations:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP)
        Recall (Sensitivity) = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
        AUC = Area Under the Receiver Operating Characteristic Curve
    
    Where:
        TP = True Positives, TN = True Negatives
        FP = False Positives, FN = False Negatives
    
    Example:
        >>> metrics = evaluate_model(model, X_test, y_test)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Print evaluation start message
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 60)
    print("EVALUATING SVM MODEL")
    print("=" * 60)
    print(f"Test samples: {X_test.shape[0]}")
    
    # -------------------------------------------------------------------------
    # Step 2: Make predictions on test data
    # -------------------------------------------------------------------------
    
    # The predict() method applies the learned decision function:
    #   ŷ = sign(f(x))
    # Where f(x) is the decision function value
    
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    
    # -------------------------------------------------------------------------
    # Step 3: Calculate accuracy
    # -------------------------------------------------------------------------
    
    # Accuracy = Number of correct predictions / Total predictions
    # Equation: Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # -------------------------------------------------------------------------
    # Step 4: Calculate precision
    # -------------------------------------------------------------------------
    
    # Precision = True Positives / (True Positives + False Positives)
    # Measures: Of all predicted positives, how many are actually positive?
    # Equation: Precision = TP / (TP + FP)
    
    precision = precision_score(y_test, y_pred, average='weighted')
    
    # -------------------------------------------------------------------------
    # Step 5: Calculate Recall (Sensitivity)
    # -------------------------------------------------------------------------
    
    # Recall (Sensitivity) = True Positives / (True Positives + False Negatives)
    # Measures: Of all actual positives, how many did we correctly identify?
    # Equation: Recall = TP / (TP + FN)
    
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # -------------------------------------------------------------------------
    # Step 6: Calculate F1-score
    # -------------------------------------------------------------------------
    
    # F1-Score = Harmonic mean of Precision and Recall
    # Equation: F1 = 2 * (Precision * Recall) / (Precision + Recall)
    # Balances precision and recall into a single metric
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # -------------------------------------------------------------------------
    # Step 7: Calculate confusion matrix
    # -------------------------------------------------------------------------
    
    # Confusion Matrix shows:
    #   [[TN, FP],
    #    [FN, TP]]
    # Where:
    #   - TN: Correctly predicted fake news
    #   - FP: Real news incorrectly predicted as fake
    #   - FN: Fake news incorrectly predicted as real
    #   - TP: Correctly predicted real news
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # -------------------------------------------------------------------------
    # Step 8: Calculate Specificity and AUC
    # -------------------------------------------------------------------------
    
    # Extract TN, FP, FN, TP
    # Note: confusion_matrix returns [[TN, FP], [FN, TP]] for binary classification
    if conf_matrix.shape == (2, 2):
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # Fallback for non-binary or edge cases
        specificity = 0.0
        
    # Calculate AUC (Area Under Curve)
    # Requires confidence scores (decision function) instead of labels
    if hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    elif hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.predict(X_test) # Fallback to binary predictions
        
    auc = roc_auc_score(y_test, y_score)
    
    # -------------------------------------------------------------------------
    # Step 9: Generate classification report
    # -------------------------------------------------------------------------
    
    # Classification report shows per-class metrics:
    #   - Precision, Recall, F1-score for each class
    #   - Support (number of samples) for each class
    
    report = classification_report(
        y_test,
        y_pred,
        target_names=['Fake News', 'Real News']
    )
    
    # -------------------------------------------------------------------------
    # Step 10: Print results
    # -------------------------------------------------------------------------
    
    print("\n" + "-" * 40)
    print("EVALUATION RESULTS")
    print("-" * 40)
    print(f"Accuracy:             {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print(f"F1-Score:             {f1:.4f}")
    print(f"ROC AUC:              {auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 11: Create and return results dictionary
    # -------------------------------------------------------------------------
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'auc_score': auc,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'predictions': y_pred,
        'y_score': y_score
    }
    
    return results


def plot_confusion_matrix(
    confusion_mat: np.ndarray,
    output_dir: str = None,
    filename: str = "confusion_matrix.png"
) -> str:
    """
    ===========================================================================
    PLOT CONFUSION MATRIX
    ===========================================================================
    
    Description:
        Creates and saves a visualization of the confusion matrix.
    
    Parameters:
        confusion_mat : numpy.ndarray
            The confusion matrix from evaluate_model().
        
        output_dir : str, optional
            Directory to save the plot.
            Default: OUTPUT_PLOTS_DIR global configuration.
        
        filename : str, optional
            Name of the output file.
            Default: "confusion_matrix.png"
    
    Returns:
        str : Path to the saved plot file.
    
    Example:
        >>> plot_path = plot_confusion_matrix(metrics['confusion_matrix'])
        >>> print(f"Plot saved to: {plot_path}")
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Resolve output directory
    # -------------------------------------------------------------------------
    
    if output_dir is None:
        output_dir = OUTPUT_PLOTS_DIR
    
    # Convert to Path and create directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Step 2: Create figure and axis
    # -------------------------------------------------------------------------
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # -------------------------------------------------------------------------
    # Step 3: Create heatmap of confusion matrix
    # -------------------------------------------------------------------------
    
    # Display the confusion matrix as a color-coded heatmap
    im = ax.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Add colorbar
    ax.figure.colorbar(im, ax=ax)
    
    # -------------------------------------------------------------------------
    # Step 4: Set labels and title
    # -------------------------------------------------------------------------
    
    # Class labels
    classes = ['Fake News', 'Real News']
    
    # Set tick labels
    ax.set(
        xticks=np.arange(confusion_mat.shape[1]),
        yticks=np.arange(confusion_mat.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title='SVM Confusion Matrix',
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    # -------------------------------------------------------------------------
    # Step 5: Add text annotations
    # -------------------------------------------------------------------------
    
    # Add the count values in each cell
    thresh = confusion_mat.max() / 2.0
    
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            ax.text(
                j,
                i,
                format(confusion_mat[i, j], 'd'),
                ha="center",
                va="center",
                color="white" if confusion_mat[i, j] > thresh else "black"
            )
    
    # -------------------------------------------------------------------------
    # Step 6: Save the figure
    # -------------------------------------------------------------------------
    
    # Adjust layout
    fig.tight_layout()
    
    # Save plot
    save_path = output_path / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Confusion matrix saved to: {save_path}")
    
    print(f"Confusion matrix saved to: {save_path}")
    
    return str(save_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    auc_score_val: float,
    output_dir: str = None,
    filename: str = "roc_curve.png"
) -> str:
    """
    ===========================================================================
    PLOT ROC CURVE
    ===========================================================================
    
    Description:
        Creates and saves the Receiver Operating Characteristic (ROC) curve.
    
    Parameters:
        y_true : numpy.ndarray
            True binary labels.
            
        y_score : numpy.ndarray
            Target scores (confidence or probability).
            
        auc_score_val : float
            Calculated Area Under Curve score.
        
        output_dir : str, optional
            Directory to save the plot.
            Default: OUTPUT_PLOTS_DIR global configuration.
        
        filename : str, optional
            Name of the output file.
            Default: "roc_curve.png"
    
    Returns:
        str : Path to the saved plot file.
        
    Example:
        >>> plot_path = plot_roc_curve(y_test, y_score, auc)
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Resolve output directory
    # -------------------------------------------------------------------------
    
    if output_dir is None:
        output_dir = OUTPUT_PLOTS_DIR
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Step 2: Calculate ROC curve
    # -------------------------------------------------------------------------
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    # -------------------------------------------------------------------------
    # Step 3: Create plot
    # -------------------------------------------------------------------------
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(
        fpr, tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (area = {auc_score_val:.2f})'
    )
    
    # Plot diagonal line (random guess)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # -------------------------------------------------------------------------
    # Step 4: Set labels and title
    # -------------------------------------------------------------------------
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Step 5: Save plot
    # -------------------------------------------------------------------------
    
    fig.tight_layout()
    save_path = output_path / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"ROC curve saved to: {save_path}")
    
    return str(save_path)


def save_model(
    model: SVC,
    output_dir: str = None,
    model_name: str = None,
    metadata: dict = None
) -> str:
    """
    ===========================================================================
    SAVE MODEL
    ===========================================================================
    
    Description:
        Saves the trained SVM model to disk for later use.
    
    Parameters:
        model : SVC
            Trained SVM classifier to save.
        
        output_dir : str, optional
            Directory to save the model.
            Default: OUTPUT_MODEL_DIR global configuration.
        
        model_name : str, optional
            Name for the model file (without extension).
            Default: "svm_model_<timestamp>"
        
        metadata : dict, optional
            Additional metadata to save with the model.
    
    Returns:
        str : Path to the saved model file.
    
    Example:
        >>> model_path = save_model(model, model_name="svm_linear_v1")
        >>> print(f"Model saved to: {model_path}")
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Resolve output directory
    # -------------------------------------------------------------------------
    
    if output_dir is None:
        output_dir = OUTPUT_MODEL_DIR
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Step 2: Generate model filename
    # -------------------------------------------------------------------------
    
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"svm_model_{timestamp}"
    
    # -------------------------------------------------------------------------
    # Step 3: Save the model
    # -------------------------------------------------------------------------
    
    model_path = output_path / f"{model_name}.joblib"
    
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    # Save using joblib (efficient for large numpy arrays)
    dump(model, model_path)
    
    print(f"Model saved to: {model_path}")
    
    # -------------------------------------------------------------------------
    # Step 4: Save metadata if provided
    # -------------------------------------------------------------------------
    
    if metadata is not None:
        metadata_path = output_path / f"{model_name}_metadata.json"
        
        # Convert numpy types to Python types for JSON serialization
        clean_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                clean_metadata[key] = value.tolist()
            elif isinstance(value, np.integer):
                clean_metadata[key] = int(value)
            elif isinstance(value, np.floating):
                clean_metadata[key] = float(value)
            else:
                clean_metadata[key] = value
        
        with open(metadata_path, 'w') as f:
            json.dump(clean_metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_path}")
    
    print("=" * 60)
    
    return str(model_path)


def load_trained_model(model_path: str) -> SVC:
    """
    ===========================================================================
    LOAD TRAINED MODEL
    ===========================================================================
    
    Description:
        Loads a previously saved SVM model from disk.
    
    Parameters:
        model_path : str
            Path to the saved model file (.joblib).
    
    Returns:
        SVC : The loaded SVM classifier.
    
    Example:
        >>> model = load_trained_model("SvmTrainedModel/svm_model.joblib")
        >>> predictions = model.predict(X_test)
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Validate path exists
    # -------------------------------------------------------------------------
    
    model_file = Path(model_path)
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # -------------------------------------------------------------------------
    # Step 2: Load and return the model
    # -------------------------------------------------------------------------
    
    print(f"Loading model from: {model_path}")
    model = load(model_file)
    print("Model loaded successfully!")
    
    return model


def predict_single(
    model: SVC,
    title: str,
    text: str,
    features_dir: str = None
) -> dict:
    """
    ===========================================================================
    PREDICT SINGLE
    ===========================================================================
    
    Description:
        Predicts whether a single news article is fake or real.
        This function handles all preprocessing (TF-IDF transformation).
    
    Parameters:
        model : SVC
            Trained SVM classifier.
        
        title : str
            Title of the news article.
        
        text : str
            Body text of the news article.
        
        features_dir : str, optional
            Path to features directory (for loading artifacts).
            Default: FEATURES_DIR global configuration.
    
    Returns:
        dict : Prediction result containing:
            - prediction : int (0=fake, 1=real)
            - label : str ("Fake News" or "Real News")
            - probability : dict (if probability enabled)
    
    Example:
        >>> result = predict_single(model, "Breaking News", "Some news text...")
        >>> print(f"Prediction: {result['label']}")
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Resolve features directory
    # -------------------------------------------------------------------------
    
    if features_dir is None:
        features_dir = FEATURES_DIR
    
    # -------------------------------------------------------------------------
    # Step 2: Load feature artifacts
    # -------------------------------------------------------------------------
    
    # Load the TF-IDF vectorizer and scaler from training
    artifacts = load_artifacts(features_dir)
    
    # Create feature config (must match training config)
    config = FeatureConfig()
    
    # -------------------------------------------------------------------------
    # Step 3: Transform the input text to features
    # -------------------------------------------------------------------------
    
    # Use transform_records to convert raw text to feature vector
    X = transform_records(
        titles=[title],
        texts=[text],
        subjects=None,
        artifacts=artifacts,
        config=config,
        scaled=USE_SCALED_FEATURES
    )
    
    # -------------------------------------------------------------------------
    # Step 4: Make prediction
    # -------------------------------------------------------------------------
    
    prediction = model.predict(X)[0]
    
    # -------------------------------------------------------------------------
    # Step 5: Get probability if available
    # -------------------------------------------------------------------------
    
    probability = None
    if hasattr(model, 'predict_proba') and model.probability:
        proba = model.predict_proba(X)[0]
        probability = {
            'fake': float(proba[0]),
            'real': float(proba[1])
        }
    
    # -------------------------------------------------------------------------
    # Step 6: Create and return result
    # -------------------------------------------------------------------------
    
    result = {
        'prediction': int(prediction),
        'label': 'Real News' if prediction == 1 else 'Fake News',
        'probability': probability
    }
    
    return result


def predict_batch(
    model: SVC,
    titles: list,
    texts: list,
    features_dir: str = None
) -> dict:
    """
    ===========================================================================
    PREDICT BATCH
    ===========================================================================
    
    Description:
        Predicts whether multiple news articles are fake or real.
        Efficient batch processing for large datasets.
    
    Parameters:
        model : SVC
            Trained SVM classifier.
        
        titles : list
            List of article titles.
        
        texts : list
            List of article body texts.
        
        features_dir : str, optional
            Path to features directory (for loading artifacts).
            Default: FEATURES_DIR global configuration.
    
    Returns:
        dict : Batch prediction results containing:
            - predictions : list of int (0=fake, 1=real)
            - labels : list of str ("Fake News" or "Real News")
            - probabilities : list of dict (if probability enabled)
    
    Example:
        >>> titles = ["Title 1", "Title 2"]
        >>> texts = ["Text 1", "Text 2"]
        >>> results = predict_batch(model, titles, texts)
        >>> print(results['labels'])
    ===========================================================================
    """
    # -------------------------------------------------------------------------
    # Step 1: Validate inputs
    # -------------------------------------------------------------------------
    
    if len(titles) != len(texts):
        raise ValueError("titles and texts must have the same length")
    
    # -------------------------------------------------------------------------
    # Step 2: Resolve features directory
    # -------------------------------------------------------------------------
    
    if features_dir is None:
        features_dir = FEATURES_DIR
    
    # -------------------------------------------------------------------------
    # Step 3: Load feature artifacts
    # -------------------------------------------------------------------------
    
    artifacts = load_artifacts(features_dir)
    config = FeatureConfig()
    
    # -------------------------------------------------------------------------
    # Step 4: Transform all texts to features
    # -------------------------------------------------------------------------
    
    X = transform_records(
        titles=titles,
        texts=texts,
        subjects=None,
        artifacts=artifacts,
        config=config,
        scaled=USE_SCALED_FEATURES
    )
    
    # -------------------------------------------------------------------------
    # Step 5: Make predictions
    # -------------------------------------------------------------------------
    
    predictions = model.predict(X)
    
    # -------------------------------------------------------------------------
    # Step 6: Get probabilities if available
    # -------------------------------------------------------------------------
    
    probabilities = None
    if hasattr(model, 'predict_proba') and model.probability:
        proba = model.predict_proba(X)
        probabilities = [
            {'fake': float(p[0]), 'real': float(p[1])}
            for p in proba
        ]
    
    # -------------------------------------------------------------------------
    # Step 7: Create labels
    # -------------------------------------------------------------------------
    
    labels = [
        'Real News' if p == 1 else 'Fake News'
        for p in predictions
    ]
    
    # -------------------------------------------------------------------------
    # Step 8: Create and return results
    # -------------------------------------------------------------------------
    
    results = {
        'predictions': [int(p) for p in predictions],
        'labels': labels,
        'probabilities': probabilities
    }
    
    return results


def train_svm_model(
    features_dir: str = None,
    output_model_dir: str = None,
    output_plots_dir: str = None,
    use_scaled: bool = None,
    kernel: str = None,
    C: float = None,
    gamma: str = None,
    degree: int = None,
    coef0: float = None,
    class_weight: str = None,
    probability: bool = None,
    random_state: int = None,
    max_iter: int = None,
    verbose: bool = None,
    model_name: str = None
) -> dict:
    """
    ===========================================================================
    TRAIN SVM MODEL (FULL PIPELINE)
    ===========================================================================
    
    Description:
        Complete training pipeline that:
        1. Loads training data
        2. Creates SVM model
        3. Trains the model
        4. Evaluates on test data
        5. Generates plots
        6. Saves the model
    
    Parameters:
        All parameters are optional. If not provided, global configurations
        are used. See individual parameters in global configuration section.
    
    Returns:
        dict : Training results containing:
            - model : SVC (trained model)
            - metrics : dict (evaluation metrics)
            - model_path : str (path to saved model)
            - plots : dict (paths to generated plots)
    
    Example:
        >>> results = train_svm_model(kernel='rbf', C=10.0)
        >>> print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    ===========================================================================
    """
    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("                    SVM TRAINING PIPELINE")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = load_training_data(
        features_dir=features_dir,
        use_scaled=use_scaled
    )
    
    # =========================================================================
    # STEP 2: CREATE MODEL
    # =========================================================================
    
    model = create_svm_model(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        class_weight=class_weight,
        probability=probability,
        random_state=random_state,
        max_iter=max_iter,
        verbose=verbose
    )
    
    # =========================================================================
    # STEP 3: TRAIN MODEL
    # =========================================================================
    
    model = train_model(model, X_train, y_train)
    
    # =========================================================================
    # STEP 4: EVALUATE MODEL
    # =========================================================================
    
    metrics = evaluate_model(model, X_test, y_test)
    
    # =========================================================================
    # STEP 5: GENERATE PLOTS
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Resolve output directories
    plots_dir = output_plots_dir if output_plots_dir else OUTPUT_PLOTS_DIR
    model_dir = output_model_dir if output_model_dir else OUTPUT_MODEL_DIR
    
    # Generate confusion matrix plot
    cm_path = plot_confusion_matrix(
        confusion_mat=metrics['confusion_matrix'],
        output_dir=plots_dir
    )
    
    # Generate ROC curve plot
    roc_path = plot_roc_curve(
        y_true=y_test,
        y_score=metrics['y_score'],
        auc_score_val=metrics['auc_score'],
        output_dir=plots_dir
    )
    
    # =========================================================================
    # STEP 6: SAVE MODEL
    # =========================================================================
    
    # Prepare metadata
    metadata = {
        'kernel': model.kernel,
        'C': model.C,
        'gamma': str(model.gamma) if isinstance(model.gamma, str) else model.gamma,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'specificity': metrics['specificity'],
        'f1_score': metrics['f1_score'],
        'auc_score': metrics['auc_score'],
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'n_features': X_train.shape[1],
        'trained_at': datetime.now().isoformat()
    }
    
    model_path = save_model(
        model=model,
        output_dir=model_dir,
        model_name=model_name,
        metadata=metadata
    )
    
    # =========================================================================
    # STEP 7: RETURN RESULTS
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("                    TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nFinal Accuracy:       {metrics['accuracy'] * 100:.2f}%")
    print(f"Final Precision:      {metrics['precision'] * 100:.2f}%")
    print(f"Final Recall (Sens):  {metrics['recall'] * 100:.2f}%")
    print(f"Final Specificity:    {metrics['specificity'] * 100:.2f}%")
    print(f"Final ROC AUC:        {metrics['auc_score']:.4f}")
    print(f"Final F1-Score:       {metrics['f1_score'] * 100:.2f}%")
    print(f"Model saved to:       {model_path}")
    print("=" * 70)
    
    results = {
        'model': model,
        'metrics': metrics,
        'model_path': model_path,
        'plots': {
            'confusion_matrix': cm_path
        }
    }
    
    return results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    """
    ===========================================================================
    MAIN EXECUTION BLOCK
    ===========================================================================
    
    This block runs when the script is executed directly from command line:
        python svm_train.py
    
    It executes the full training pipeline using global configurations.
    
    To customize training, either:
        1. Modify the global configuration variables at the top of this file
        2. Import this module and call train_svm_model() with custom parameters
    ===========================================================================
    """
    
    # Run the complete training pipeline
    results = train_svm_model()