"""
External Evaluation Script: ISOT-trained model -> WELFake_Dataset.csv

Assumptions:
- You already generated ISOT features/artifacts in "features_out/" using your feature_pipeline.
- WELFake CSV has columns: title, text, label (0=fake, 1=real) and is already correct.

What this script does:
1) Loads ISOT features (train/test) and trains a baseline model
2) Calculates performance metrics on the ISOT dataset
3) Loads + transforms WELFake using the feature_pipeline helper
4) Calculates performance metrics on the WELFake dataset
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from features_pipeline import (
    load_feature_matrices,
    load_welfake_external_eval,
)

# =========================
# Config you may adjust
# =========================
FEATURES_OUT_DIR = "Output/features_out"
WELFAKE_CSV_PATH = "External_Datasets/WELFake_Dataset.csv"

# Use scaled features for LR/SVM
USE_SCALED_FOR_MODEL = True

# Apply the same scaling for WELFake transform (recommended for LR/SVM)
SCALE_EXTERNAL = True

# Limit rows if your RAM is tight (set None for full dataset)
WELFAKE_LIMIT = 20000  # e.g., 20000


def _print_metrics(name: str, y_true, y_pred, y_prob=None) -> None:
    print(f"\n========== {name} ==========")
    print(f"Samples: {len(y_true)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        print(f"\nROC-AUC : {auc:.4f}")


def main() -> None:
    # -----------------------------
    # 1) Load the feature and test matrices based on ISOT training set
    # -----------------------------
    X_train, X_test, y_train, y_test = load_feature_matrices(
        FEATURES_OUT_DIR, scaled=USE_SCALED_FOR_MODEL
    )

    # -----------------------------
    # 2) Train a model
    # -----------------------------
    model = LogisticRegression(max_iter=3000, n_jobs=-1)
    model.fit(X_train, y_train)

    # -----------------------------
    # 3) Evaluate on ISOT test set
    # -----------------------------
    y_isot_pred = model.predict(X_test)
    y_isot_prob = model.predict_proba(X_test)[:, 1]
    _print_metrics("ISOT (Test)", y_test, y_isot_pred, y_isot_prob)

    # -----------------------------
    # 4) Load + transform WELFake
    # -----------------------------
    X_wel, y_wel = load_welfake_external_eval(
        welfake_csv_path=WELFAKE_CSV_PATH,
        features_out_dir=FEATURES_OUT_DIR,
        scaled=SCALE_EXTERNAL,
        limit=WELFAKE_LIMIT,
    )

    # -----------------------------
    # 5) Evaluate on WELFake
    # -----------------------------
    y_wel_pred = model.predict(X_wel)
    y_wel_prob = model.predict_proba(X_wel)[:, 1]
    _print_metrics("WELFake (External)", y_wel, y_wel_pred, y_wel_prob)


if __name__ == "__main__":
    main()