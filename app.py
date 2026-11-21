from flask import Flask, request, jsonify
from Predict import predict_location

app = Flask(__name__)

@app.get("/")
def home():
    return """
    <h2>AI-Based Hidden Accident Blackspot Discovery</h2>
    <p>POST JSON to /predict with latitude and longitude.</p>
    <p>Example: {"latitude": 13.0827, "longitude": 80.2707}</p>
    """

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            return jsonify({"error": "Latitude/longitude out of range"}), 400
        result = predict_location(latitude, longitude)
        return jsonify(result)
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"error": f"Invalid input: {error}"}), 400
    except FileNotFoundError:
        return jsonify({"error": "Model not found. Run clean_data.py and train.py first."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
