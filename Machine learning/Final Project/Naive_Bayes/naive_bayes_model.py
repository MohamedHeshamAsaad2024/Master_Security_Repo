# 1. Standard library imports
import os
import sys
import json

# 2. Third-party library imports
import joblib  # Used for saving and loading models
import pandas as pd  # Used for data manipulation (CSV reading)
import numpy as np   # Used for numerical operations
from typing import Dict, Any, Optional, List

from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             classification_report, roc_curve, auc, precision_recall_curve, 
                             average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
                             make_scorer, roc_auc_score)

import matplotlib.pyplot as plt

def specificity_score(y_true, y_pred):
    """Calculates Specificity (True Negative Rate)"""
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0

# 4. Custom project imports
# We add the preprocessing directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_preprocessing_and_cleanup')))
import features_pipeline

class NaiveBayesClassifierWrapper:
    """
    A 'Wrapper' class that makes it easy to work with Naive Bayes models.
    It bundles training, tuning, saving, and predicting into one simple tool.
    """
    
    def __init__(self, features_dir: str):
        """
        INITIALIZATION: This runs when you first create the 'nb' object.
        It sets up the storage for models and loads the 'ingredients' prepared by the pipeline.
        """
        self.features_dir = features_dir
        
        # A dictionary to store the three different types of Naive Bayes models
        self.models = {
            'BNB': None,  # Bernoulli
            'CNB': None,  # Complement
            'MNB': None   # Multinomial
        }
        
        # Placeholders for the best model we find after tuning
        self.best_model_type = None
        self.best_model = None
        
        # Placeholders for storage of scores (Accuracy, F1, etc.) and best settings
        self.metrics = {}
        self.best_params = {}
        
        # Step 1: Load the 'Artifacts' (the saved Tfidf vocabulary) from the features directory
        self.artifacts = features_pipeline.load_artifacts(features_dir)
        
        # Step 2: Load the configuration file (config.json) to know exactly how the data was cleaned
        config_path = os.path.join(features_dir, "artifacts", "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            from features_pipeline import FeatureConfig
            # Recreate the FeatureConfig object using the saved data
            self.config = FeatureConfig(**config_dict)

    def train(self, param_grids: Dict[str, Dict[str, List[Any]]], model_types: Optional[List[str]] = None):
        """
        TRAINING & TUNING: This is where the brain 'learns' from the data.
        It tries different flavors (MNB, BNB, CNB) and different settings (alpha) to find the best one.
        """
        # If no specific models were requested, we train all three by default
        if model_types is None:
            model_types = ['BNB', 'CNB', 'MNB']
            
        # Step 1: Load the numeric data (the matrices) from the disk
        # We use scaled=False because Naive Bayes works better with unscaled counts
        X_train, X_test, y_train, y_test = features_pipeline.load_feature_matrices(self.features_dir, scaled=False)
        y_train = y_train.ravel() # Flatten the labels into a simple 1D list
        y_test = y_test.ravel()

        # A map to easily access the scikit-learn classes
        model_classes = {
            'BNB': BernoulliNB,
            'CNB': ComplementNB,
            'MNB': MultinomialNB
        }

        # Step 2: Set up 'Cross-Validation'
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        overall_scores = {}

        # Define multi-metric scoring for plotting
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'sensitivity': 'recall_weighted', # Sensitivity = Recall
            'specificity': make_scorer(specificity_score),
            'f1': 'f1_weighted',
            'auc': 'roc_auc'
        }

        # Step 3: Loop through each type of model (BNB, CNB, MNB)
        for m_type in model_types:
            if m_type not in model_classes:
                continue
                
            m_class = model_classes[m_type]
            print(f"Training and tuning {m_type}...")
            
            # Step 4: Run 'Grid Search' with Multi-Metric Scoring
            grid = GridSearchCV(
                m_class(), 
                param_grids.get(m_type, {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}), 
                cv=skf, 
                scoring=scoring,
                refit='f1', # We still pick the best based on F1 for refitting
                n_jobs=-1 
            )
            grid.fit(X_train, y_train)

            # --- PLOTTING CAPABILITY ---
            self._plot_metrics(grid.cv_results_, m_type)
            
            # Step 5: Save the best version of this specific model type
            best_m = grid.best_estimator_
            self.models[m_type] = best_m
            self.best_params[m_type] = grid.best_params_
            
            # Step 6: Test the model on the 'Test Set'
            y_pred = best_m.predict(X_test)
            y_prob = best_m.predict_proba(X_test)[:, 1] if hasattr(best_m, "predict_proba") else None
            
            # --- ADVANCED DIAGNOSTICS PLOTS ---
            self._plot_diagnostic_curves(y_test, y_pred, y_prob, m_type)

            m_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'sensitivity': recall_score(y_test, y_pred, average='weighted'),
                'specificity': specificity_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'auc': roc_auc_score(y_test, y_prob) if y_prob is not None else 0
            }
            self.metrics[m_type] = m_metrics
            
            # Step 7: Selection criterion - We use the F1-Score as requested
            # This provides a balance between Precision and Recall
            overall_scores[m_type] = m_metrics['f1']
            
            print(f"{m_type} Best Params: {grid.best_params_}")
            print(f"{m_type} Metrics: {m_metrics}")

        # Step 8: Choose the best-performing model out of all the types tested based on F1
        if overall_scores:
            self.best_model_type = max(overall_scores, key=overall_scores.get)
            self.best_model = self.models[self.best_model_type]
            print(f"\nOverall Best Model: {self.best_model_type} with F1-Score: {overall_scores[self.best_model_type]:.4f}")
        
        return self.best_params, self.metrics

    def _plot_metrics(self, cv_results: Dict, model_type: str):
        """
        Generates and saves a plot of Accuracy, Precision, Recall, and F1 across different alphas.
        """
        alphas = cv_results['param_alpha'].data
        # Handle cases where alpha might not be a simple list (e.g. if other params were tuned)
        # For this project, we primarily tune alpha.
        
        plt.figure(figsize=(10, 6))
        
        metrics_to_plot = {
            'Accuracy': 'mean_test_accuracy',
            'Precision': 'mean_test_precision',
            'Sensitivity': 'mean_test_sensitivity',
            'Specificity': 'mean_test_specificity',
            'F1-Score': 'mean_test_f1',
            'AUC-ROC': 'mean_test_auc'
        }
        
        for label, key in metrics_to_plot.items():
            plt.plot(alphas, cv_results[key], marker='o', label=label)
            
        plt.title(f"{model_type} Performance vs Alpha")
        plt.xlabel("Alpha")
        plt.ylabel("Score")
        plt.xscale('log') # Better visualization for alpha values spanning orders of magnitude
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        
        # Save plot
        plot_dir = os.path.join(os.path.dirname(__file__), "Plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{model_type}_tuning.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved for {model_type} alpha tuning at {plot_path}")

    def _plot_diagnostic_curves(self, y_true, y_pred, y_prob, model_type: str):
        """
        Generates and saves ROC Curve, Precision-Recall Curve, and Confusion Matrix.
        """
        plot_dir = os.path.join(os.path.dirname(__file__), "Plots", model_type)
        os.makedirs(plot_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake (0)', 'Real (1)'])
        disp.plot(cmap='Blues', values_format='d')
        acc = accuracy_score(y_true, y_pred)
        plt.title(f"Confusion Matrix - {model_type}\nAccuracy: {acc*100:.2f}%")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{model_type}_confusion_matrix.png"))
        plt.close()

        # 2. ROC Curve
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5000)')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_type}')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f"{model_type}_roc_curve.png"))
            plt.close()

            # 3. Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            avg_prec = average_precision_score(y_true, y_prob)
            
            plt.figure(figsize=(8, 8))
            plt.step(recall, precision, color='darkblue', lw=2, where='post', label=f'PR Curve (Avg Precision = {avg_prec:.4f})')
            # Baseline is often the fraction of positives
            baseline = sum(y_true) / len(y_true)
            plt.axhline(y=baseline, color='red', linestyle='--', label=f'Random Classifier (Baseline = {baseline:.4f})')
            plt.xlabel('Recall (Sensitivity)')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_type}')
            plt.legend(loc="lower left")
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f"{model_type}_pr_curve.png"))
            plt.close()
            
        print(f"Diagnostic plots saved for {model_type} in {plot_dir}")

    def predict_single(self, title: str, text: str, subject: Optional[str] = None, 
                       model_type: str = 'best', parameters: Optional[Dict] = None) -> int:
        """
        SINGLE PREDICTION: Used for checking one article (like in a GUI).
        """
        # Step 1: Use the Pipeline to turn raw words into numbers (the Digital Matrix)
        X = features_pipeline.transform_records(
            titles=[title],
            texts=[text],
            subjects=[subject] if subject else None,
            artifacts=self.artifacts,
            config=self.config,
            scaled=False # Always False for NB
        )
        
        # Step 2: Get the model we want to use (either the 'Best' one or a specific type)
        target_model = self._get_model(model_type, parameters)
        
        # Step 3: Ask the model for the answer (0=Fake, 1=Real)
        prediction = target_model.predict(X)[0]
        return int(prediction)

    def predict_csv(self, csv_path: str, model_type: str = 'best', 
                    parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        BATCH PREDICTION: Reads a fast list of articles from a CSV file.
        """
        # Step 1: Load the CSV file using Pandas
        df = pd.read_csv(csv_path)
        
        # Step 2: Convert the entire file's text into one giant digital matrix
        X = features_pipeline.transform_records(
            titles=df['title'].fillna("").astype(str).tolist(),
            texts=df['text'].fillna("").astype(str).tolist(),
            subjects=df['subject'].fillna("").astype(str).tolist() if 'subject' in df.columns else None,
            artifacts=self.artifacts,
            config=self.config,
            scaled=False
        )
        
        # Step 3: Use the model to predict all articles at once
        target_model = self._get_model(model_type, parameters)
        predictions = target_model.predict(X)
        
        result = {'predictions': predictions.tolist()}
        
        # Step 4: If the CSV file already has labels (like for testing), calculate accuracy
        label_col = 'label' if 'label' in df.columns else ('target' if 'target' in df.columns else None)
        
        if label_col:
            y_true = df[label_col].values
            result['metrics'] = {
                'accuracy': accuracy_score(y_true, predictions),
                'precision': precision_score(y_true, predictions, average='weighted'),
                'recall': recall_score(y_true, predictions, average='weighted'),
                'f1': f1_score(y_true, predictions, average='weighted')
            }
        
        return result

    def _get_model(self, model_type: str, parameters: Optional[Dict]):
        """
        Helper to retrieve or train a model with specific parameters.
        """
        model_classes = {'BNB': BernoulliNB, 'CNB': ComplementNB, 'MNB': MultinomialNB}
        
        if parameters:
            # If parameters are provided, we must train a new model with them
            if model_type == 'best':
                model_type = self.best_model_type or 'BNB'
            
            print(f"Training {model_type} with custom parameters: {parameters}...")
            # Load training data
            X_train, _, y_train, _ = features_pipeline.load_feature_matrices(self.features_dir, scaled=False)
            y_train = y_train.ravel()
            
            m = model_classes[model_type](**parameters)
            m.fit(X_train, y_train)
            return m

        if model_type == 'best':
            if not self.best_model:
                raise ValueError("Model not trained yet. Call train() first.")
            return self.best_model
        
        m = self.models.get(model_type)
        if not m:
            # If not already trained/tuned, try to fit a default one
            print(f"Model {model_type} not found in cache. Fitting default...")
            X_train, _, y_train, _ = features_pipeline.load_feature_matrices(self.features_dir, scaled=False)
            y_train = y_train.ravel()
            m = model_classes[model_type]()
            m.fit(X_train, y_train)
            self.models[model_type] = m
            
        return m

    def save_models(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for m_type, m_obj in self.models.items():
            if m_obj:
                joblib.dump(m_obj, os.path.join(save_dir, f"nb_{m_type}.joblib"))
        
        metadata = {
            'best_model_type': self.best_model_type,
            'best_params': self.best_params,
            'metrics': self.metrics
        }
        joblib.dump(metadata, os.path.join(save_dir, "nb_metadata.joblib"))
        print(f"Models and metadata saved to {save_dir}")

    def load_models(self, save_dir: str):
        for m_type in self.models.keys():
            path = os.path.join(save_dir, f"nb_{m_type}.joblib")
            if os.path.exists(path):
                self.models[m_type] = joblib.load(path)
        
        metadata_path = os.path.join(save_dir, "nb_metadata.joblib")
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.best_model_type = metadata.get('best_model_type')
            self.best_params = metadata.get('best_params')
            self.metrics = metadata.get('metrics')
            if self.best_model_type:
                self.best_model = self.models[self.best_model_type]
        print(f"Models loaded from {save_dir}")
