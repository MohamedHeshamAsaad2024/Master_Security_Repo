# SVM Fake News Classification

Complete Support Vector Machine (SVM) implementation for fake news classification with training, testing, and GUI modules.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Module](#training-module)
3. [Testing Module](#testing-module)
4. [Multi-Configuration Training](#multi-configuration-training)
5. [GUI Module](#gui-module)
6. [API Reference](#api-reference)
7. [Configuration Guide](#configuration-guide)

---

## Quick Start

### Prerequisites

```bash
# Install required packages
pip install numpy pandas scikit-learn matplotlib joblib tqdm
```

### Train a Model

```bash
cd "SVM"
python svm_train.py
```

### Test the Model

```bash
cd "SVM"
python svm_test.py
```

### Launch GUI

```bash
cd "GUI"
python gui.py
```

---

## Training Module

### File: `svm_train.py`

The training module provides complete SVM training with configurable kernels and hyperparameters.

### Command Line Usage

```bash
# Train with default configuration (Linear kernel)
python svm_train.py

# Output:
# - Model saved to: SvmTrainedModel/
# - Plots saved to: SvmTrainedOutput/
```

### Python API Usage

```python
from svm_train import train_svm_model, predict_single

# Train with custom configuration
results = train_svm_model(
    kernel='rbf',
    C=10.0,
    gamma='scale'
)

# Get accuracy
print(f"Accuracy: {results['metrics']['accuracy']:.4f}")

# Predict on new article
result = predict_single(
    model=results['model'],
    title="Breaking News",
    text="Article content here..."
)
print(f"Prediction: {result['label']}")
```

### Configuration Parameters

Edit the global variables at the top of `svm_train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `KERNEL_TYPE` | `"linear"` | Kernel: `linear`, `rbf`, `poly`, `sigmoid` |
| `C_REGULARIZATION` | `1.0` | Regularization strength |
| `GAMMA` | `"scale"` | Kernel coefficient (for non-linear) |
| `DEGREE` | `3` | Polynomial degree (for poly kernel) |
| `USE_SCALED_FEATURES` | `True` | Use StandardScaler normalized features |
| `ENABLE_PROBABILITY` | `True` | Enable probability estimates (for ROC) |
| `OUTPUT_MODEL_DIR` | `SvmTrainedModel` | Model save directory |
| `OUTPUT_PLOTS_DIR` | `SvmTrainedOutput` | Plots save directory |

---

## Testing Module

### File: `svm_test.py`

The testing module provides evaluation with live progress bar and metrics.

### Command Line Usage

```bash
# Test on WELFake dataset (default)
python svm_test.py

# Shows:
# - Live progress bar
# - Real-time accuracy, precision, recall, F1
# - Final confusion matrix
```

### Python API Usage

```python
from svm_test import test_on_csv, test_single_article

# Test on CSV file
results = test_on_csv(
    csv_path="path/to/dataset.csv",
    batch_size=500
)

# Test single article
result = test_single_article(
    title="News Title",
    text="Article content..."
)
print(f"Prediction: {result['label']}")
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OUTPUT_TEST_DIR` | `SvmTestOutput` | Results save directory |
| `BATCH_SIZE` | `500` | Samples per batch |
| `LIMIT_SAMPLES` | `None` | Limit test samples |
| `SHOW_PROGRESS_BAR` | `True` | Display progress bar |
| `SHOW_LIVE_METRICS` | `True` | Show live metrics |
| `SAVE_PREDICTIONS` | `True` | Save predictions CSV |

---

## Multi-Configuration Training

### File: `MultiConfig.py`

Automated script to train and test multiple SVM configurations, generating comparison reports.

### Run All Configurations

```bash
python MultiConfig.py
```

This will:
1. Train 10 different SVM configurations (linear, RBF, poly, sigmoid)
2. Test each on WELFake dataset
3. Save all outputs to timestamped folder
4. Generate comparison plot and report

### Regenerate Report from Existing Results

```bash
python MultiConfig.py MultiConfig_Results_YYYYMMDD_HHMMSS
```

Use this to regenerate comparison plot and report without re-training.

### Output Structure

```
MultiConfig_Results_YYYYMMDD_HHMMSS/
├── Linear_C1/
│   ├── SvmTrainedModel/      # Model files
│   ├── SvmTrainedOutput/     # Training plots
│   ├── SvmTestOutput/        # Test results
│   └── config_summary.json
├── RBF_scale/
│   └── ...
├── comparison_plot.png       # Visual comparison
├── comparison_report.txt     # Text summary
└── all_results.json          # All data in JSON
```

### Configurations Tested

| Name | Kernel | C | Gamma | Notes |
|------|--------|---|-------|-------|
| Linear_C1 | linear | 1.0 | - | Default |
| Linear_C10 | linear | 10.0 | - | Higher C |
| Linear_C0.1 | linear | 0.1 | - | Lower C |
| RBF_scale | rbf | 1.0 | scale | Auto gamma |
| RBF_0.1 | rbf | 1.0 | 0.1 | Fixed gamma |
| Poly_deg3 | poly | 1.0 | scale | Degree 3 |
| Poly_deg4 | poly | 1.0 | scale | Degree 4 |
| Sigmoid | sigmoid | 1.0 | scale | Neural-like |
| Linear_balanced | linear | 1.0 | - | Balanced weights |

---

## GUI Module

### File: `GUI/gui.py`

Tkinter-based graphical interface for training and testing.

### Launch

```bash
cd GUI
python gui.py
```

### Features

- **Train Tab**: Configure and train models
- **Test Tab**: Test single articles or CSV files
- **Model Selection**: Choose from trained models
- **Live Display**: Real-time metrics during testing
- **Plots**: View confusion matrix and ROC curve

---

## API Reference

### Training Functions

| Function | Description |
|----------|-------------|
| `train_svm_model()` | Complete training pipeline |
| `create_svm_model()` | Create SVM with config |
| `train_model()` | Fit model to data |
| `evaluate_model()` | Evaluate on test data |
| `save_model()` | Save model to disk |
| `load_trained_model()` | Load saved model |
| `predict_single()` | Predict single article |
| `predict_batch()` | Predict multiple articles |

### Testing Functions

| Function | Description |
|----------|-------------|
| `test_on_csv()` | Test on CSV dataset |
| `test_on_welfake()` | Test on WELFake dataset |
| `test_single_article()` | Test single article |
| `run_live_test()` | Run with progress bar |
| `save_test_results()` | Save results and plots |

---

## Configuration Guide

### Choosing a Kernel

| Kernel | Best For | Speed | Notes |
|--------|----------|-------|-------|
| `linear` | Text classification, sparse data | Fast | Works well with TF-IDF |
| `rbf` | Complex patterns | Slow | Highest accuracy potential |
| `poly` | Polynomial relationships | Medium | Use degree 2-3 |
| `sigmoid` | Neural-like behavior | Medium | Less stable |

### Tuning C (Regularization)

| Value | Effect |
|-------|--------|
| 0.01 | High regularization, simple model |
| 1.0 | Balanced (recommended start) |
| 100.0 | Low regularization, complex model |

### Example Configurations

**High Accuracy (Slower)**
```python
KERNEL_TYPE = "rbf"
C_REGULARIZATION = 10.0
GAMMA = "scale"
```

**Fast Training**
```python
KERNEL_TYPE = "linear"
C_REGULARIZATION = 1.0
```

**Balanced**
```python
KERNEL_TYPE = "linear"
C_REGULARIZATION = 1.0
CLASS_WEIGHT = "balanced"
```

---

## Output Files

### Training Output

```
SvmTrainedModel/
├── svm_model_YYYYMMDD_HHMMSS.joblib    # Trained model
└── svm_model_YYYYMMDD_HHMMSS_metadata.json

SvmTrainedOutput/
├── confusion_matrix.png
└── roc_curve.png
```

### Testing Output

```
SvmTestOutput/
├── test_*_metrics.json           # Metrics summary
├── test_*_confusion_matrix.png   # Confusion matrix plot
├── test_*_predictions.csv        # Individual predictions
└── test_*_classification_report.txt
```

---

## Troubleshooting

### Model Not Found

Run training first:
```bash
python svm_train.py
```

### Memory Error

Reduce dataset size:
```python
LIMIT_SAMPLES = 50000
```

### Slow Training

Use linear kernel:
```python
KERNEL_TYPE = "linear"
```

---

## Integration with Other Models

See `GUI/MODEL_INTEGRATION.md` for instructions on integrating:
- Logistic Regression
- Naive Bayes
- XGBoost
