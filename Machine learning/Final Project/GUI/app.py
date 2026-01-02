
import sys
import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------
# Path Construction & Imports
# ---------------------------------------------------------
# Define paths relative to this script
GUI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESSING_DIR = os.path.join(PROJECT_ROOT, "Data_preprocessing_and_cleanup")
MODEL_DIR = os.path.join(PROJECT_ROOT, "Naive Bayes", "models")
FEATURES_OUT_DIR = os.path.join(PREPROCESSING_DIR, "Output", "features_out")

# Add paths to sys.path
if PREPROCESSING_DIR not in sys.path:
    sys.path.append(PREPROCESSING_DIR)

NAIVE_BAYES_DIR = os.path.dirname(MODEL_DIR) # c:\...\Naive Bayes
if NAIVE_BAYES_DIR not in sys.path:
    sys.path.append(NAIVE_BAYES_DIR)

try:
    from features_pipeline import transform_records, FeatureConfig, load_artifacts
    # Import custom model class to ensure joblib can deserialize it
    from naive_bayes_model import BernoulliNB, MultinomialNB, ComplementNB
except ImportError:
    try:
        from feature_pipeline import transform_records, FeatureConfig, load_artifacts
        from naive_bayes_model import BernoulliNB, MultinomialNB, ComplementNB
    except ImportError:
        print("Error: Could not import 'features_pipeline' or 'naive_bayes_model'. Check directories.")
        sys.exit(1)

# ---------------------------------------------------------
# App Setup
# ---------------------------------------------------------
app = Flask(__name__)

# Global variables for artifacts
model = None
artifacts = None
config = None

def load_resources():
    global model, artifacts, config
    print("Loading resources...")
    
    # 1. Load Model
    model_path = os.path.join(MODEL_DIR, "best_naive_bayes.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training script first.")
    model = joblib.load(model_path)
    print(f"Model loaded: {type(model).__name__}")

    # 2. Load Feature Artifacts
    if not os.path.exists(FEATURES_OUT_DIR):
        raise FileNotFoundError(f"Features directory not found at {FEATURES_OUT_DIR}.")
    artifacts = load_artifacts(FEATURES_OUT_DIR)
    
    # 3. Load Config
    config_path = os.path.join(FEATURES_OUT_DIR, "artifacts", "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = FeatureConfig(**json.load(f))
    
    print("All resources loaded successfully.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        title = data.get('title', '')
        text = data.get('text', '')
        model_type = data.get('model_type', 'naive_bayes')

        if not title and not text:
            return jsonify({'error': 'No content provided'}), 400

        # Model Routing
        if model_type != 'naive_bayes':
            if model_type in ['logistic_regression', 'svm', 'xgboost']:
                 return jsonify({'error': f'{model_type} is not implemented yet. Please select Naive Bayes.'}), 501
            else:
                 return jsonify({'error': f'Unknown model type: {model_type}'}), 400

        # Transform Input
        # Note: transform_records handles cleaning strings, so raw input is fine
        X_input = transform_records([title], [text], None, artifacts, config, scaled=False)

        # Predict
        prediction_cls = model.predict(X_input)[0] # 0 or 1
        probabilities = model.predict_proba(X_input)[0] # [prob_0, prob_1]
        
        confidence = float(max(probabilities))
        label = "REAL" if prediction_cls == 1 else "FAKE"
        
        # Explainability (Top Features)
        # We want to see which words in THIS document contributed to the score.
        # Simple approach: Get nonzero feature indices in input, and look up their weights in the model.
        
        # 1. Get nonzero indices from sparse vector
        _, col_indices = X_input.nonzero()
        
        # 2. Get feature names
        tfidf = artifacts['tfidf']
        all_feature_names = np.array(tfidf.get_feature_names_out())
        
        # 3. Get Model Weights (Log Probabilities) for the predicted class
        # 3. Get Model Weights (Log Probabilities) for the predicted class
        # Note: The custom Naive Bayes model is not a Pipeline, so we access attributes directly.
        # We trained on the full feature set, so feature_names match 1:1.
        
        feature_names = all_feature_names
        selected_col_indices = col_indices  

        # Get log probs for the predicted class
        class_log_probs = model.feature_log_prob_[prediction_cls]
        
        # Find which present words have high weights for this class
        # We iterate over the words actually present in the input (selected_col_indices)
        explain_list = []
        for idx in selected_col_indices:
            word = feature_names[idx]
            weight = class_log_probs[idx]
            # Heuristic: simple log prob value. Closer to 0 (less negative) = higher probability
            explain_list.append({"word": word, "weight": float(weight)})
            
        # Sort by weight (highest/least negative first)
        explain_list.sort(key=lambda x: x['weight'], reverse=True)
        top_features = explain_list[:5] # Top 5 words

        return jsonify({
            'label': label,
            'confidence': confidence,
            'top_features': top_features
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_resources()
    app.run(debug=True, port=5000)
