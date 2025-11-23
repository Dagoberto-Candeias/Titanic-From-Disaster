from flask import Flask, request, jsonify
import logging
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optionally load the trained model pipeline (for future prediction endpoints)
model_pipeline = None
try:
    model_path = "output/models/best_model_pipeline.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_pipeline = pickle.load(f)
        logger.info("✅ Trained model pipeline loaded successfully.")
    else:
        logger.warning("⚠️  Model pipeline not found at output/models/best_model_pipeline.pkl. Prediction endpoints will not be available.")
except Exception as e:
    logger.error(f"❌ Error loading model pipeline: {e}")

@app.route('/welcome', methods=['GET'])
def welcome():
    # Log request metadata: method and path
    logger.info(f"Request received: {request.method} {request.path}")
    # Return JSON response with welcome message
    return jsonify({"message": "Welcome to the Titanic API"})

if __name__ == '__main__':
    app.run(debug=True)
