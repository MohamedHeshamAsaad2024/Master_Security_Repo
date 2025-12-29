# Feature Pipeline Module Documentation

## Overview

`feature_pipeline` is a reusable feature engineering and preprocessing module for building Fake News detection models using the Kaggle **Fake & Real News Dataset (ISOT)**.

It transforms raw `Fake.csv` and `True.csv` files into machine-learning-ready sparse feature matrices using:

* Text cleaning
* TF-IDF vectorization
* Optional subject encoding
* Standardization for LR / SVM
* Leakage-safe artifact storage for deployment
* **External dataset compatibility for cross-dataset evaluation (e.g., WELFake)**

---

## Generated Outputs

All outputs are written into the specified `out_dir`.

| File                   | Description                                                            |
| ---------------------- | ---------------------------------------------------------------------- |
| `X_train_unscaled.npz` | Sparse TF-IDF training matrix (used for MultinomialNB & XGBClassifier) |
| `X_test_unscaled.npz`  | Sparse TF-IDF test matrix                                              |
| `X_train_scaled.npz`   | Standardized training matrix (used for LR & SVM)                       |
| `X_test_scaled.npz`    | Standardized test matrix                                               |
| `y_train.csv`          | Training labels (`0=fake`, `1=real`)                                   |
| `y_test.csv`           | Test labels                                                            |

### Artifacts Folder

| Artifact             | Purpose                                    |
| -------------------- | ------------------------------------------ |
| `tfidf.joblib`       | Fitted `TfidfVectorizer`                   |
| `subject_ohe.joblib` | One-Hot encoder for `subject` (if enabled) |
| `scaler.joblib`      | Fitted `StandardScaler`                    |
| `config.json`        | Snapshot of experiment configuration       |
| `split_info.json`    | Train/test split metadata                  |
| `stats.json`         | Feature statistics & class distributions   |

---

## FeatureConfig Reference

`FeatureConfig` controls every preprocessing step. It is immutable and fully reproducible.

| Parameter                 | Description                                |
| ------------------------- | ------------------------------------------ |
| `random_state`            | Seed controlling dataset splitting         |
| `test_size`               | Portion reserved for testing               |
| `include_subject`         | Append subject as one-hot encoded features |
| `drop_date`               | Always removes date column                 |
| `fill_missing_with_empty` | Replaces NaN values                        |
| `drop_exact_duplicates`   | Removes identical articles                 |
| `lowercase`               | Lowercases all text                        |
| `remove_urls_emails`      | Removes URLs/emails                        |
| `normalize_whitespace`    | Cleans repeated spaces                     |
| `ngram_min`, `ngram_max`  | Word n-gram range                          |
| `max_features`            | Vocabulary size                            |
| `min_df`, `max_df`        | Token filtering thresholds                 |
| `stop_words`              | English stopword removal                   |
| `sublinear_tf`            | Log-scaled term frequency                  |
| `strip_accents`           | Unicode normalization                      |
| `produce_scaled`          | Enables StandardScaler                     |
| `scaler_with_mean`        | Must be False for sparse matrices          |

---

## Main API

### 1. Build Features (ISOT)

```python
from feature_pipeline import FeatureConfig, build_and_save_features

cfg = FeatureConfig(include_subject=False)
build_and_save_features("Fake.csv", "True.csv", out_dir="features_out", config=cfg)
```

---

### 2. Load Feature Matrices

```python
from feature_pipeline import load_feature_matrices

# For MultinomialNB / XGBClassifier
X_train, X_test, y_train, y_test = load_feature_matrices("features_out", scaled=False)

# For LR / SVM
X_train_s, X_test_s, y_train, y_test = load_feature_matrices("features_out", scaled=True)
```

---

### 3. Load Artifacts for Deployment

```python
from feature_pipeline import load_artifacts

artifacts = load_artifacts("features_out")
```

---

### 4. Transform New Samples (Inference)

```python
from feature_pipeline import transform_records, FeatureConfig

cfg = FeatureConfig(include_subject=False)

X_new = transform_records(
    titles=["Breaking: New policy announced"],
    texts=["The government today issued a new directive..."],
    subjects=None,
    artifacts=artifacts,
    config=cfg,
    scaled=False
)
```

---

## External Dataset Evaluation – WELFake

The module provides a built-in helper for evaluating your ISOT-trained models on **WELFake_Dataset.csv**.

### 5. Load WELFake for External Evaluation

```python
from feature_pipeline import load_welfake_external_eval

X_wel, y_wel = load_welfake_external_eval(
    welfake_csv_path="External_Datasets/WELFake_Dataset.csv",
    features_out_dir="features_out",
    scaled=True,        # Must match model type
    limit=10000         # Optional RAM limiter
)
```

This function:

* Loads the saved TF-IDF + scaler artifacts from ISOT training
* Cleans WELFake using the same preprocessing rules
* Applies TF-IDF + optional scaling
* Returns ready-to-predict matrices `(X_wel, y_wel)`

This ensures **no leakage, no retraining, and true cross-dataset generalization testing**.

---

## Model Usage Mapping

| Model               | Input Matrix     |
| ------------------- | ---------------- |
| MultinomialNB       | `*_unscaled.npz` |
| Logistic Regression | `*_scaled.npz`   |
| SVM                 | `*_scaled.npz`   |
| XGBClassifier       | `*_unscaled.npz` |

---

## Auxilary Files

| Parameter                 | Description                                |
| ------------------------- | ------------------------------------------ |
| `Module_Test.py`          | This script generates machine-learning-ready TF-IDF feature matrices from the ISOT Fake/Real News dataset using the feature_pipeline module and reloads them for modeling. It then performs strict sanity checks to ensure the feature matrices and labels are consistent, valid, and safe to use for training classification models.         |
| `EDA.py`               | This script performs basic EDA for the ISOT dataset               |
| `Datasets_Evaluation.py`         | This scipt shows a direct use of the generated features, model training and performance evaluation over the ISOT dataset and the WELFake dataset |

---

## Design Guarantees

* No data leakage
* Cross-dataset reproducibility
* External dataset compatibility
* Memory-safe sparse matrices
* Fully configurable preprocessing
* Production-ready inference pipeline