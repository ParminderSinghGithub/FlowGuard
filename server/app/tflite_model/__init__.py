# tflite_model/__init__.py
# Main package initialization
import os

# Define constants used across the package
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
H5_MODEL_PATH = os.path.join(MODEL_DIR, 'traffic_lstm_model.h5')
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, 'traffic_lstm_model.tflite')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)