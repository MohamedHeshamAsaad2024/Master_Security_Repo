import os
from naive_bayes_model import NaiveBayesClassifierWrapper

def test_nb_flow():
    # Paths (adjust according to your environment)
    features_dir = r"c:\Master\Repos\Master_Security_Repo\Machine learning\Final Project\Data_preprocessing_and_cleanup\Output\features_out"
    save_dir = r"c:\Master\Repos\Master_Security_Repo\Machine learning\Final Project\Naive Bayes\Models"
    external_csv = r"c:\Master\Repos\Master_Security_Repo\Machine learning\Final Project\Data_preprocessing_and_cleanup\External_Datasets\WELFake_Dataset.csv"
    
    nb = NaiveBayesClassifierWrapper(features_dir)
    
    # 1. Example: Training only BNB
    print("--- 1. Training BNB only ---")
    param_grids = {
        'BNB': {'alpha': [0.1, 1.0, 10.0]}
    }
    best_params, metrics = nb.train(param_grids, model_types=['BNB'])
    
    # 2. Example: Predict with any classifier and specific parameters
    # This demonstrates choosing MNB even if it wasn't tuned in train(), 
    # and providing custom parameters.
    print("\n--- 2. Predicting with MNB and custom parameters (alpha=2.0) ---")
    title = "Trump says Russia probe is 'witch hunt'"
    text = "WASHINGTON (Reuters) - U.S. President Donald Trump on Thursday again called the investigation..."
    pred = nb.predict_single(title, text, model_type='MNB', parameters={'alpha': 2.0})
    print(f"Prediction result: {'Real' if pred == 1 else 'Fake'} ({pred})")
    
    # 3. Example: Evaluate on External CSV (WELFake)
    if os.path.exists(external_csv):
        print(f"\n--- 3. Evaluating on External CSV: {os.path.basename(external_csv)} ---")
        # We use BNB (the tuned one) for this batch prediction
        result = nb.predict_csv(external_csv, model_type='BNB')
        
        print("\nFinal Results for WELFake:")
        if 'metrics' in result:
            m = result['metrics']
            print(f"Accuracy:  {m['accuracy']:.4f}")
            print(f"Precision: {m['precision']:.4f}")
            print(f"Recall:    {m['recall']:.4f}")
            print(f"F1 Score:  {m['f1']:.4f}")
        else:
            print("No labels found in CSV, only predictions generated.")
    else:
        print(f"\nExternal CSV not found at: {external_csv}")

    # Save models for persistence
    nb.save_models(save_dir)

if __name__ == "__main__":
    test_nb_flow()
