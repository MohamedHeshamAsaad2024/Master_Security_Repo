
import requests
import pandas as pd
import time
import os

# Configuration
API_URL = "http://localhost:5000"
TEST_CSV = "test_batch.csv"

def create_sample_csv():
    print("[1] Creating sample CSV...")
    data = {
        "title": [
            "Breaking: Scientific Discovery in Antarctica",
            "You won't believe what this politician said!",
            "Global Economy Shows Steady Growth",
            "Confirmed: Alien Base Found on Moon",
            "Local Library to Host Book Fair"
        ],
        "text": [
            "Scientists have discovered a new species of bacteria in the subglacial lakes of Antarctica. The finding suggests life can thrive in extreme conditions.",
            "CLICK HERE to see the shocking truth that the mainstream media is hiding from you. Confirmed by anonymous sources!",
            "The World Bank report indicated that most major economies are recovering well from recent downturns, with inflation stabilizing.",
            "Leaked documents show a massive underground facility on the dark side of the moon. Government has been hiding this for decades. BREAKING NEWS.",
            "The downtown library will host its annual book fair this weekend, featuring local authors and rare editions."
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv(TEST_CSV, index=False)
    print(f"    Sample CSV created: {TEST_CSV}")

def test_batch_process():
    print("\n[2] Testing Batch Prediction API...")
    if not os.path.exists(TEST_CSV):
        print("    Error: Sample CSV missing.")
        return

    with open(TEST_CSV, 'rb') as f:
        files = {'file': f}
        data = {'algorithm': 'best_naive_bayes'}
        try:
            response = requests.post(f"{API_URL}/predict_batch", files=files, data=data)
            if response.status_code == 200:
                results = response.json()
                print(f"    Success! Received {len(results)} predictions.")
                for r in results:
                    print(f"    - {r['prediction']}: {r['title'][:40]}...")
            else:
                print(f"    Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"    Connection error: {e}")

def test_training():
    print("\n[3] Testing Model Tuning API...")
    try:
        # Trigger training
        response = requests.post(f"{API_URL}/train")
        print(f"    Trigger result: {response.json()['status']}")
        
        # Poll status
        for _ in range(20): # 60 seconds max
            status_resp = requests.get(f"{API_URL}/train/status")
            status = status_resp.json()
            if not status['ongoing']:
                print("    Training complete!")
                # print(f"    Result Summary: {status['last_result'][:200]}...")
                return
            print("    Still training...")
            time.sleep(3)
        print("    Timeout waiting for training.")
    except Exception as e:
        print(f"    Connection error: {e}")

if __name__ == "__main__":
    create_sample_csv()
    # Note: app.py must be running for these tests to work
    print("\nStarting tests... (Ensure app.py is running on localhost:5000)")
    test_batch_process()
    test_training()
    
    # Cleanup
    if os.path.exists(TEST_CSV):
        os.remove(TEST_CSV)
