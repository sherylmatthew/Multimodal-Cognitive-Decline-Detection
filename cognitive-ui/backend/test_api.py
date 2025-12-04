import requests
import json
from typing import Dict, Any

def make_prediction(data: Dict[str, Any]) -> None:
    try:
        r = requests.post("http://127.0.0.1:5000/predict", json=data)
        
        try:
            response_json = r.json()
            if not r.ok:
                print(f"Server error (Status {r.status_code}):")
                print(f"Error message: {response_json.get('error', 'No error message provided')}")
                print(f"Status: {response_json.get('status', 'unknown')}")
                return
                
            print("Prediction results:")
            print(json.dumps(response_json, indent=2))
            
        except json.JSONDecodeError:
            print(f"Failed to decode JSON response. Status code: {r.status_code}")
            print(f"Raw response: {r.text}")
            
    except requests.exceptions.ConnectionError:
        print("Could not connect to server. Make sure the Flask server is running on http://127.0.0.1:5000")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")


data = {
    "speech_path": "",
    "text_content": "The patient speaks slower and pauses often.",
    "handwriting_path": "",
    "visual_data": {},
    "physio_data": {}
}

if __name__ == "__main__":
    make_prediction(data)