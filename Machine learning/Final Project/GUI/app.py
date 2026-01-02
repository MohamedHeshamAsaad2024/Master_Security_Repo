
import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
import subprocess
import threading
from flask import Flask, request, jsonify, render_template, send_file
import io

# ---------------------------------------------------------
# Path Construction & Imports
# ---------------------------------------------------------
GUI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESSING_DIR = os.path.join(PROJECT_ROOT, "Data_preprocessing_and_cleanup")
MODEL_DIR = os.path.join(PROJECT_ROOT, "Naive Bayes", "models")
FEATURES_OUT_DIR = os.path.join(PREPROCESSING_DIR, "Output", "features_out")
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "Naive Bayes", "train_tune_naive_bayes.py")

# Add paths to sys.path
if PREPROCESSING_DIR not in sys.path:
    sys.path.append(PREPROCESSING_DIR)

NAIVE_BAYES_DIR = os.path.join(PROJECT_ROOT, "Naive Bayes")
if NAIVE_BAYES_DIR not in sys.path:
    sys.path.append(NAIVE_BAYES_DIR)

try:
    from features_pipeline import transform_records, FeatureConfig, load_artifacts
    from naive_bayes_model import BernoulliNB, MultinomialNB, ComplementNB
except ImportError as e:
    print(f"Error: {e}. Check directories.")
    sys.exit(1)

# ---------------------------------------------------------
# App Setup
# ---------------------------------------------------------
app = Flask(__name__)

# Global storage
models_dict = {}
artifacts = None
config = None
training_status = {"ongoing": False, "last_result": None}

def load_resources():
    global artifacts, config, models_dict
    print("Loading resources...")
    
    # 1. Load Model Variants
    variants = ["mnb", "cnb", "bnb", "best_naive_bayes"]
    for v in variants:
        path = os.path.join(MODEL_DIR, f"{v}.joblib")
        if os.path.exists(path):
            models_dict[v] = joblib.load(path)
            print(f"Loaded variant: {v}")
    
    if not models_dict:
        print("Warning: No models found. GUI will need training first.")

    # 2. Load Feature Artifacts
    if os.path.exists(FEATURES_OUT_DIR):
        artifacts = load_artifacts(FEATURES_OUT_DIR)
        config_path = os.path.join(FEATURES_OUT_DIR, "artifacts", "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = FeatureConfig(**json.load(f))
    
    print("All initial resources loaded.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        title = data.get('title', '')
        text = data.get('text', '')
        
        # Dynamic Config
        algo_key = data.get('algorithm', 'best_naive_bayes') # mnb, cnb, bnb, best_naive_bayes
        alpha = float(data.get('alpha', 0.01))
        fit_prior = bool(data.get('fit_prior', True))
        norm = bool(data.get('norm', False))
        binarize = float(data.get('binarize', 0.0))

        if algo_key in ['logistic_regression', 'svm', 'xgboost']:
            return jsonify({
                'label': 'N/A',
                'confidence': 0,
                'algorithm_used': algo_key,
                'top_features': [],
                'status': 'Model integration in progress'
            })

        if algo_key not in models_dict:
            return jsonify({'error': f'Model {algo_key} not loaded'}), 404

        model = models_dict[algo_key]
        
        # Apply Hot-Swap Configuration
        # We use a copy if it's a global model to avoid state pollution, 
        # but for simple single-user app, direct update is fine.
        model.update_params(alpha=alpha, fit_prior=fit_prior)
        if hasattr(model, 'norm'): model.norm = norm
        if hasattr(model, 'binarize'): model.binarize = binarize

        # Determine if scaling is needed (for placeholders LR/SVM)
        is_scaled = algo_key in ['logistic_regression', 'svm']
        
        # Transform
        X_input = transform_records([title], [text], None, artifacts, config, scaled=is_scaled)

        # Predict
        prediction_cls = int(model.predict(X_input)[0])
        probabilities = model.predict_proba(X_input)[0].tolist()
        
        confidence = float(max(probabilities))
        label = "REAL" if prediction_cls == 1 else "FAKE"
        
        # Top Features for explainability
        _, col_indices = X_input.nonzero()
        tfidf = artifacts['tfidf']
        all_feature_names = np.array(tfidf.get_feature_names_out())
        
        class_log_probs = model.feature_log_prob_[prediction_cls]
        explain_list = []
        for idx in col_indices:
            explain_list.append({"word": all_feature_names[idx], "weight": float(class_log_probs[idx])})
            
        explain_list.sort(key=lambda x: x['weight'], reverse=True)
        top_features = explain_list[:5]

        return jsonify({
            'label': label,
            'confidence': confidence,
            'top_features': top_features,
            'algorithm_used': type(model).__name__
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Handles CSV upload for large scale prediction.
    Expects columns 'title' and 'text'.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    algo_key = request.form.get('algorithm', 'best_naive_bayes')
    
    # Validate Algorithm
    if algo_key not in ['best_naive_bayes', 'mnb', 'cnb', 'bnb', 'logistic_regression', 'svm', 'xgboost']:
        return jsonify({'error': f'Invalid algorithm: {algo_key}'}), 400

    if algo_key in ['logistic_regression', 'svm', 'xgboost']:
        return jsonify({
            'results': [],
            'summary': {'total': 0, 'status': 'Implementation in progress'},
            'status': 'Multi-algorithm batch support coming soon'
        })
        
    if algo_key not in models_dict:
        return jsonify({'error': 'Model not loaded'}), 404
        
    try:
        df = pd.read_csv(file)
        if 'title' not in df.columns or 'text' not in df.columns:
            return jsonify({'error': 'CSV must have "title" and "text" columns'}), 400
        
        model = models_dict[algo_key]
        
        # Extract subject if available
        subjects = df['subject'].tolist() if 'subject' in df.columns else None
        
        # Determine if scaling is needed
        is_scaled = algo_key in ['logistic_regression', 'svm']
        
        # Transform all records at once (Scaling prediction)
        X_all = transform_records(
            df['title'].tolist(), 
            df['text'].tolist(), 
            subjects, 
            artifacts, 
            config, 
            scaled=is_scaled
        )
        
        # Batch Predict
        preds = model.predict(X_all)
        df['prediction'] = ["REAL" if p == 1 else "FAKE" for p in preds]
        
        # Calculate Summary Analytics
        total = len(df)
        real_count = int(np.sum(preds == 1))
        fake_count = int(np.sum(preds == 0))
        
        summary = {
            "total": total,
            "real_count": real_count,
            "fake_count": fake_count,
            "real_percentage": round((real_count / total) * 100, 2) if total > 0 else 0,
            "fake_percentage": round((fake_count / total) * 100, 2) if total > 0 else 0
        }
        
        # Return as JSON
        results = df[['title', 'prediction']].to_dict(orient='records')
        return jsonify({
            "results": results,
            "summary": summary
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def trigger_train():
    global training_status
    if training_status["ongoing"]:
        return jsonify({'status': 'Training already in progress'}), 400
    
    def run_training():
        global training_status
        training_status["ongoing"] = True
        try:
            # Parse dynamic grids from request
            data = request.get_json() or {}
            cmd = [sys.executable, TRAIN_SCRIPT]
            
            # Append optional grids
            if data.get('mnb_grid'):
                cmd.extend(["--mnb_grid", json.dumps(data['mnb_grid'])])
            if data.get('cnb_grid'):
                cmd.extend(["--cnb_grid", json.dumps(data['cnb_grid'])])
            if data.get('bnb_grid'):
                cmd.extend(["--bnb_grid", json.dumps(data['bnb_grid'])])
                
            result = subprocess.run(cmd, capture_output=True, text=True)
            training_status["last_result"] = result.stdout
            load_resources() # Reload models after training
        except Exception as e:
            training_status["last_result"] = f"Error: {e}"
        finally:
            training_status["ongoing"] = False

    thread = threading.Thread(target=run_training)
    thread.start()
    return jsonify({'status': 'Training started in background'})

@app.route('/train/status', methods=['GET'])
def get_train_status():
    return jsonify(training_status)

if __name__ == '__main__':
    load_resources()
    app.run(debug=True, port=5000)
