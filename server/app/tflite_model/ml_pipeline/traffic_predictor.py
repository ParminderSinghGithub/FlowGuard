# tflite_model/ml_pipeline/traffic_predictor.py
from collections import deque
from datetime import timedelta
import numpy as np
import tensorflow as tf
import joblib
import os

class TrafficPredictor:
    def __init__(self, tflite_path=None, scaler_path='scaler.pkl'):
        # Set default paths relative to this file if not provided
        if tflite_path is None:
            from tflite_model import TFLITE_MODEL_PATH
            tflite_path = TFLITE_MODEL_PATH
            
        if not os.path.isabs(scaler_path):
            scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), scaler_path)
        
        # Load model and scaler
        self.model = tf.lite.Interpreter(model_path=tflite_path)
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        self.scaler = joblib.load(scaler_path)
        self.feature_window = {}  # {road_segment_id: deque of recent features}
    
    def update_features(self, new_data):
        """Update real-time features for each road segment"""
        for segment in new_data:
            segment_id = segment['road_segment_id']
            if segment_id not in self.feature_window:
                self.feature_window[segment_id] = deque(maxlen=10)  # Match model input length
            
            # Normalize speed properly
            normalized = self.scaler.transform(
                np.array(
                    [[segment['current_speed']]]
                )
            )[0][0]
            
            # Prepare spatial features
            spatial_features = [
                segment.get('merge_points', 0),
                segment.get('capacity', 1.0),
                segment.get('free_flow_ratio', 1.0)
            ]
            
            # Track hour of day as normalized feature (0-1)
            hour_of_day = segment['timestamp'].hour / 24.0
            
            # Store all features
            self.feature_window[segment_id].append({
                'normalized_speed': normalized,
                'timestamp': segment['timestamp'],
                'hour': hour_of_day,
                'dow': segment['timestamp'].weekday() / 7.0,  # Normalize day of week
                'spatial_features': spatial_features
            })
    
    def predict_congestion(self, segment_id, steps_ahead=1):
        """Predict congestion for a road segment N steps ahead"""
        if segment_id not in self.feature_window:
            return None
        
        window = list(self.feature_window[segment_id])
        if len(window) < 3:  # Need minimal history
            return None
        
        # Prepare temporal input tensor - match expected shape (1, 10, 5)
        temporal_data = np.zeros((10, 5), dtype=np.float32)
        
        # Fill available data - latest points first
        for i, w in enumerate(reversed(window)):
            if i >= 10:  # Only use last 10 points
                break
                
            # Basic features: [speed, hour, dow, 0, 0]
            # (last 2 zeros are placeholders for merge_proximity and capacity_util)
            temporal_data[i, 0] = w['normalized_speed']
            temporal_data[i, 1] = w['hour']
            temporal_data[i, 2] = w['dow']
            
            # Leave remaining features as zeros - could be enhanced later
        
        # Add speed trend features
        if len(window) >= 3:
            speeds = [w['normalized_speed'] for w in window[-3:]]
            for i in range(min(len(window), 10)):
                temporal_data[i, 3] = np.mean(speeds)  # rolling mean
                
                # Simple trend (difference)
                if i > 0 and i < len(window):
                    temporal_data[i, 4] = speeds[0] - speeds[-1]  # trend
        
        temporal_input = np.expand_dims(temporal_data, axis=0)
        
        # Prepare spatial input - match expected shape (1, 3)
        spatial_input = np.array([window[-1]['spatial_features']], dtype=np.float32)
        
        # Run prediction with both inputs
        self.model.set_tensor(self.input_details[0]['index'], temporal_input)
        self.model.set_tensor(self.input_details[1]['index'], spatial_input)
        self.model.invoke()
        
        # Get both outputs - speed prediction and bottleneck risk
        speed_prediction = self.model.get_tensor(self.output_details[0]['index'])
        bottleneck_risk = self.model.get_tensor(self.output_details[1]['index'])
        
        # Convert normalized speed back to actual values
        actual_speed = self.scaler.inverse_transform(speed_prediction.reshape(-1, 1))[0][0]
        
        return {
            'predicted_speed': actual_speed,
            'bottleneck_risk': float(bottleneck_risk[0][0])
        }
