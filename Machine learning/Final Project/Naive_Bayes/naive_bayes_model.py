import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Add parent directory and pipeline directory to path to import features_pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_preprocessing_and_cleanup')))
import features_pipeline

class NaiveBayesClassifierWrapper:
    """
    A wrapper class for Naive Bayes models (BNB, CNB, MNB) with training, 
    tuning, and prediction APIs, integrated with the project feature pipeline.
    """
    
    def __init__(self, features_dir: str):
        self.features_dir = features_dir
        self.models = {
            'BNB': None,
            'CNB': None,
            'MNB': None
        }
        self.best_model_type = None
        self.best_model = None
        self.metrics = {}
        self.best_params = {}
        
        # Load pipeline artifacts
        self.artifacts = features_pipeline.load_artifacts(features_dir)
        # Load config used for feature extraction
        # features_pipeline saves config to artifacts/config.json
        config_path = os.path.join(features_dir, "artifacts", "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            from features_pipeline import FeatureConfig
            self.config = FeatureConfig(**config_dict)

    def train(self, param_grids: Dict[str, Dict[str, List[Any]]], model_types: Optional[List[str]] = None):
        """
        Trains and tunes requested Naive Bayes models using 5-fold CV.
        Selects the best overall model based on the average of F1, Accuracy, Precision, and Recall.
        
        Args:
            param_grids: Dictionary where keys are 'BNB', 'CNB', 'MNB' and values are param_grid dictionaries.
            model_types: List of model types to train (default: ['BNB', 'CNB', 'MNB']).
        """
        if model_types is None:
            model_types = ['BNB', 'CNB', 'MNB']
            
        X_train, X_test, y_train, y_test = features_pipeline.load_feature_matrices(self.features_dir, scaled=False)
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        model_classes = {
            'BNB': BernoulliNB,
            'CNB': ComplementNB,
            'MNB': MultinomialNB
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        overall_scores = {}

        for m_type in model_types:
            if m_type not in model_classes:
                continue
                
            m_class = model_classes[m_type]
            print(f"Training and tuning {m_type}...")
            
            # GridSearchCV uses f1_weighted for internal selection
            grid = GridSearchCV(
                m_class(), 
                param_grids.get(m_type, {'alpha': [0.1, 0.5, 1.0, 2.0]}), 
                cv=skf, 
                scoring='f1_weighted',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            
            best_m = grid.best_estimator_
            self.models[m_type] = best_m
            self.best_params[m_type] = grid.best_params_
            
            # Evaluate on test set to get all metrics
            y_pred = best_m.predict(X_test)
            m_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            self.metrics[m_type] = m_metrics
            
            # Selection criterion: average of all four metrics
            avg_score = (m_metrics['accuracy'] + m_metrics['precision'] + 
                         m_metrics['recall'] + m_metrics['f1']) / 4
            overall_scores[m_type] = avg_score
            
            print(f"{m_type} Best Params: {grid.best_params_}")
            print(f"{m_type} Metrics: {m_metrics} (Avg Score: {avg_score:.4f})")

        # Choose overall best based on average score
        if overall_scores:
            self.best_model_type = max(overall_scores, key=overall_scores.get)
            self.best_model = self.models[self.best_model_type]
            print(f"\nOverall Best Model: {self.best_model_type} with Avg Score: {overall_scores[self.best_model_type]:.4f}")
        
        return self.best_params, self.metrics

    def predict_single(self, title: str, text: str, subject: Optional[str] = None, 
                       model_type: str = 'best', parameters: Optional[Dict] = None) -> int:
        """
        Predicts the label for a single news instance.
        If parameters are provided, it fits a model with those parameters first.
        """
        X = features_pipeline.transform_records(
            titles=[title],
            texts=[text],
            subjects=[subject] if subject else None,
            artifacts=self.artifacts,
            config=self.config,
            scaled=False
        )
        
        target_model = self._get_model(model_type, parameters)
        prediction = target_model.predict(X)[0]
        return int(prediction)

    def predict_csv(self, csv_path: str, model_type: str = 'best', 
                    parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Predicts labels for a CSV file and calculates metrics if label column exists.
        """
        df = pd.read_csv(csv_path)
        
        # transform_records expects lists of strings
        X = features_pipeline.transform_records(
            titles=df['title'].fillna("").astype(str).tolist(),
            texts=df['text'].fillna("").astype(str).tolist(),
            subjects=df['subject'].fillna("").astype(str).tolist() if 'subject' in df.columns else None,
            artifacts=self.artifacts,
            config=self.config,
            scaled=False
        )
        
        target_model = self._get_model(model_type, parameters)
        predictions = target_model.predict(X)
        
        result = {'predictions': predictions.tolist()}
        
        # If ground truth exists, return metrics
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
