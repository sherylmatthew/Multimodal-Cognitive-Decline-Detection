from flask import Flask, request, jsonify
from cognitive_system import InferenceSystem

app = Flask(__name__)
inference = None

try:
    inference = InferenceSystem(model_path='./models/')
except Exception as e:
    print(f"Error initializing inference system: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if inference is None:
        return jsonify({
            "error": "Model not initialized properly",
            "status": "error"
        }), 500
        
    try:
        data = request.json
        if not data:
            return jsonify({
                "error": "No data provided",
                "status": "error"
            }), 400
            
        results = inference.analyze_patient(data)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)