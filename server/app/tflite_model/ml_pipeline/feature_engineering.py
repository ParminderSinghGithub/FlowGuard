# tflite_model/ml_pipeline/feature_engineering.py
from collections import deque
import numpy as np

class StreamFeatureGenerator:
    def __init__(self, window_size=10):
        self.feature_windows = {}
        self.window_size = window_size
        
    def update_features(self, processed_records):
        """Update features for multiple road segments"""
        for record in processed_records:
            seg_id = record['road_segment_id']
            
            if seg_id not in self.feature_windows:
                self.feature_windows[seg_id] = {
                    'speed': deque(maxlen=self.window_size),
                    'features': deque(maxlen=self.window_size),
                    'timestamps': deque(maxlen=self.window_size)
                }
            
            window = self.feature_windows[seg_id]
            window['speed'].append(record['normalized_speed'])
            window['timestamps'].append(record['timestamp'])
            
            # Calculate temporal features
            features = {
                'current_speed': record['normalized_speed'],
                'hour': record['timestamp'].hour,
                'dow': record['timestamp'].weekday(),
                'merge_proximity': self._calc_merge_proximity(record),
                'speed_trend': self._calc_speed_trend(window['speed']),
                'capacity_utilization': record['current_speed'] / record.get('capacity', 1.0)
            }
            
            window['features'].append(features)
        
        return self.feature_windows
    
    def _calc_merge_proximity(self, record):
        """Calculate distance to next merge point"""
        merge_points = record.get('merge_points', [])
        if not merge_points:
            return 0.0
        
        geometry = record.get('geometry', None)
        if not geometry or not hasattr(geometry, 'get'):
            return 0.0
            
        try:
            current_pos = (geometry.get('x', 0), geometry.get('y', 0))
            return min(
                np.linalg.norm(np.array(current_pos) - np.array(merge))
                for merge in merge_points
            )
        except Exception:
            return 0.0
    
    def _calc_speed_trend(self, speed_window):
        """Calculate linear trend of last speeds"""
        if len(speed_window) < 2:
            return 0.0
        try:
            x = np.arange(len(speed_window))
            slope = np.polyfit(x, list(speed_window), 1)[0]
            return slope
        except Exception:
            return 0.0
    
    def get_model_input_features(self, segment_id):
        """Extract features in format suitable for model input"""
        if segment_id not in self.feature_windows:
            return None
        
        window = self.feature_windows[segment_id]
        if len(window['features']) < 3:  # Need minimal history
            return None
        
        # Prepare temporal features matrix (sequence data)
        # Format: [timesteps, features] where features are:
        # [current_speed, hour/24, dow/7, merge_proximity, capacity_util]
        temporal_features = np.zeros((self.window_size, 5), dtype=np.float32)
        
        for i, feature_dict in enumerate(list(window['features'])):
            if i >= self.window_size:
                break
                
            temporal_features[i, 0] = feature_dict['current_speed']
            temporal_features[i, 1] = feature_dict['hour'] / 24.0  # Normalize hour
            temporal_features[i, 2] = feature_dict['dow'] / 7.0    # Normalize day of week
            temporal_features[i, 3] = feature_dict['merge_proximity']
            temporal_features[i, 4] = feature_dict['capacity_utilization']
        
        # Prepare spatial features vector
        # [merge_proximity, capacity, free_flow_ratio]
        latest = list(window['features'])[-1]
        spatial_features = np.array([
            latest['merge_proximity'],
            latest['capacity_utilization'],
            latest.get('free_flow_ratio', 1.0)
        ], dtype=np.float32)
        
        return {
            'temporal': temporal_features,
            'spatial': spatial_features
        }
