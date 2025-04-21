from django.apps import apps
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import os
import warnings

class RealTimeDataProcessor:
    def __init__(self, scaler_path=None):
        # Initialize with default scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit([[0], [100]])  # Default range 0-100 km/h
        
        # Try to load scaler if path provided
        if scaler_path and os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception as e:
                warnings.warn(f"Could not load scaler: {e}. Using default scaler.")
        
        # Road network features (optional)
        self.road_network = {}

    def process_record(self, data_point):
        """Process single TrafficData record"""
        # Get speeds with defaults
        current_speed = data_point.current_speed
        free_flow_speed = data_point.free_flow_speed if data_point.free_flow_speed > 0 else current_speed
        
        # Use TrafficData's own ID as the segment identifier
        road_segment_id = str(data_point.id)
        
        # Process the data
        processed = {
            'timestamp': data_point.timestamp,
            'road_segment_id': road_segment_id,
            'current_speed': current_speed,
            'normalized_speed': self.scaler.transform([[current_speed]])[0][0],
            'free_flow_ratio': current_speed / free_flow_speed if free_flow_speed else 1.0,
            'is_bottleneck': data_point.is_bottleneck,
            'confidence': data_point.confidence_score,
            'latitude': float(data_point.latitude),
            'longitude': float(data_point.longitude),
            'road_type': data_point.road_type,
            'location': data_point.location
        }
        
        return processed