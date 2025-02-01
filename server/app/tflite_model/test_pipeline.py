import os
import pandas as pd
import numpy as np
import tensorflow as tf
from ml_pipeline.data_processing import load_and_clean_data
from ml_pipeline.feature_engineering import generate_time_series_features
from ml_pipeline.convert_to_tflite import convert_to_tflite

# Test File Configuration
CSV_FILE = "traffic_data.csv"
H5_MODEL_PATH = "traffic_lstm_model.h5"
TFLITE_MODEL_PATH = "traffic_lstm_model.tflite"

def test_pipeline():
    print("Starting Pipeline Testing...")

    # Step 1: Load and clean data
    df, scaler = load_and_clean_data(CSV_FILE)
    print("Data Cleaning Test Passed.")

    # Step 2: Perform feature engineering
    df = generate_time_series_features(df)
    print("Feature Engineering Test Passed.")

    # Step 3: Test TensorFlow Lite conversion
    convert_to_tflite(H5_MODEL_PATH, TFLITE_MODEL_PATH)
    assert os.path.exists(TFLITE_MODEL_PATH), "TFLite model file should exist."
    print("TFLite Conversion Test Passed.")

    print("Pipeline Test Completed Successfully.")

if __name__ == "__main__":
    test_pipeline()
