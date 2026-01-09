from joblib import load
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import seaborn as sns
import sys
import os
# Add the folder containing features_pipeline.py to Python path
folder_path = r"D:\Big Data\ML - Final Project\Master_Security_Repo\Machine learning\Final Project\Data_preprocessing_and_cleanup"
sys.path.append(folder_path)

from features_pipeline import load_welfake_external_eval

WELFAKE_DATASET_PATH = r"D:\Big Data\ML - Final Project\Master_Security_Repo\Machine learning\Final Project\Data_preprocessing_and_cleanup\External_Datasets\WELFake_Dataset.csv"

FEATURES_DIR = r'D:\Big Data\ML - Final Project\Master_Security_Repo\Machine learning\Final Project\Data_preprocessing_and_cleanup\Output\features_out'
model_path = r'D:\Big Data\ML - Final Project\Master_Security_Repo\Machine learning\Final Project\XG Boost\XgbTrainedModel\xgboost_gridsearch_20260107_021537.joblib'
USE_SCALED_FEATURES = True

folder_path = r'D:\Big Data\ML - Final Project\Master_Security_Repo\Machine learning\Final Project\XG Boost\XgbTrainedModel'
if os.path.exists(folder_path):
    print("Files in XgbTrainedModel folder:")
    for file in os.listdir(folder_path):
        print(f"  - {file}")
else:
    print(f"Folder does not exist: {folder_path}")
    # -------------------------------------------------------------------------
    # Step 1: Print header
    # -------------------------------------------------------------------------
    
print("\n" + "*" * 70)
print("*        CROSS-DATASET EVALUATION: WELFake Dataset")
print("*" * 70)
print(f"Dataset: {WELFAKE_DATASET_PATH}")
print("Using load_welfake_external_eval() API from features_pipeline.py")

# -------------------------------------------------------------------------
# Load the trained model
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Step 1: Validate path exists
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Step 2: Load and return the model
# -------------------------------------------------------------------------

print(f"Loading model from: {model_path}")
model = load(model_path)
print("Model loaded successfully!")

    
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
    limit=10000
)

print(f"Total samples: {X_wel.shape[0]}")
print(f"Feature dimensions: {X_wel.shape[1]}")
print(f"Class distribution: fake={np.sum(y_wel == 0)}, real={np.sum(y_wel == 1)}")
print("=" * 60)


# Make predictions
y_wel_pred = model.predict(X_wel)
y_wel_pred_proba = model.predict_proba(X_wel)[:, 1]

# ============================================================================
# STEP 5: CALCULATE METRICS
# ============================================================================
accuracy = accuracy_score(y_wel, y_wel_pred)
precision = precision_score(y_wel, y_wel_pred)
recall = recall_score(y_wel, y_wel_pred)
f1 = f1_score(y_wel, y_wel_pred)

print("\n" + "=" * 60)
print("EVALUATION METRICS")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("=" * 60)

print("\nClassification Report:")
print(classification_report(y_wel, y_wel_pred, target_names=['Fake', 'Real']))

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix
cm = confusion_matrix(y_wel, y_wel_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'], 
            ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
axes[0, 0].set_xlabel('Predicted Label')
axes[0, 0].set_ylabel('True Label')


# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_wel, y_wel_pred_proba)
roc_auc = auc(fpr, tpr)

axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve', fontweight='bold')
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(True, alpha=0.3)

# 3. Prediction Distribution
axes[1, 0].hist(y_wel_pred_proba, bins=20, alpha=0.7, color='green', edgecolor='black')
axes[1, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Predicted Probability (Real News)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Prediction Probability Distribution', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 4. Performance Metrics Bar Chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]

axes[1, 1].bar(metrics, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
axes[1, 1].set_ylim([0, 1])
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Performance Metrics', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')


# Add value labels on bars
for i, v in enumerate(values):
    axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Save the figure
save_path = r'D:\Big Data\ML - Final Project\Master_Security_Repo\Machine learning\Final Project\XG Boost\welfake_evaluation.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {save_path}")

print("\nEvaluation complete!")