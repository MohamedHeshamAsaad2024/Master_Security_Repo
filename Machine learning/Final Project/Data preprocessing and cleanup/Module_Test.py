# -----------------------------
# Imports
# -----------------------------
from features_pipeline import FeatureConfig, build_and_save_features, load_feature_matrices
import numpy as np
from scipy.sparse import issparse

# -----------------------------
# Demo Use to generate features
# -----------------------------
# Create a Feature configuration object
cfg = FeatureConfig()

# Perform the features pre-processing
build_and_save_features("Internal_Dataset/Fake.csv", "Internal_Dataset/True.csv", out_dir="Output/features_out", config=cfg)

# Load the train and test matrices for model application
X_train, X_test, y_train, y_test = load_feature_matrices("Output/features_out", scaled=False)

# -----------------------------
# Sanity Checks
# -----------------------------
def sanity_check_ready_for_training(X_train, X_test, y_train, y_test) -> None:
    # 1) Type checks
    assert issparse(X_train), "X_train must be a scipy sparse matrix (CSR/CSC)."
    assert issparse(X_test), "X_test must be a scipy sparse matrix (CSR/CSC)."
    assert isinstance(y_train, np.ndarray), "y_train must be a numpy array."
    assert isinstance(y_test, np.ndarray), "y_test must be a numpy array."

    # 2) Shape consistency
    assert X_train.shape[0] == y_train.shape[0], "X_train rows must match y_train length."
    assert X_test.shape[0] == y_test.shape[0], "X_test rows must match y_test length."
    assert X_train.shape[1] == X_test.shape[1], "Train/Test must have same number of features."

    # 3) Label validity
    y_train_u = set(np.unique(y_train).tolist())
    y_test_u = set(np.unique(y_test).tolist())
    assert y_train_u.issubset({0, 1}), f"Unexpected labels in y_train: {y_train_u}"
    assert y_test_u.issubset({0, 1}), f"Unexpected labels in y_test: {y_test_u}"

    # 4) Feature validity: no NaNs or infs in sparse data
    if X_train.nnz > 0:
        assert np.isfinite(X_train.data).all(), "X_train contains NaN/inf."
    if X_test.nnz > 0:
        assert np.isfinite(X_test.data).all(), "X_test contains NaN/inf."

    # 5) Basic health checks
    assert X_train.nnz > 0, "X_train is empty (no non-zero entries)."
    assert X_test.nnz > 0, "X_test is empty (no non-zero entries)."

    print("Data is ready for model training.")
    print(f"X_train shape: {X_train.shape}, nnz={X_train.nnz}")
    print(f"X_test shape : {X_test.shape}, nnz={X_test.nnz}")
    print(f"y_train distribution: {np.bincount(y_train, minlength=2)} (fake=0, real=1)")
    print(f"y_test distribution : {np.bincount(y_test, minlength=2)} (fake=0, real=1)")

sanity_check_ready_for_training(X_train, X_test, y_train, y_test)