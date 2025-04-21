import numpy as np
import tensorflow as tf
from queue import PriorityQueue
from collections import defaultdict
import os
from datetime import timedelta, datetime
import math
from .feature_engineering import StreamFeatureGenerator

class RouteOptimizer:
    def __init__(self, tflite_model=None):
        self.route_graph = self._build_route_graph()
        self.model = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.feature_generator = StreamFeatureGenerator(window_size=10)

        if tflite_model:
            self._load_model(tflite_model)

    def _load_model(self, tflite_path):
        """Load TFLite model and allocate tensors"""
        if not os.path.exists(tflite_path):
            raise FileNotFoundError(f"Model file not found: {tflite_path}")

        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _build_route_graph(self):
        """Build road network graph structure (undirected)"""
        graph = {
            'segment_1': {
                'geometry': {'x': 30.911972, 'y': 75.853222},
                'length': 500,
                'connects_to': ['segment_2', 'segment_3'],
                'historical_speed': 40.0
            },
            'segment_2': {
                'geometry': {'x': 30.915000, 'y': 75.855000},
                'length': 800,
                'connects_to': ['segment_4'],
                'historical_speed': 35.0
            },
            'segment_3': {
                'geometry': {'x': 30.910000, 'y': 75.850000},
                'length': 600,
                'connects_to': ['segment_5'],
                'historical_speed': 45.0
            },
            'segment_4': {
                'geometry': {'x': 30.920000, 'y': 75.860000},
                'length': 1000,
                'connects_to': ['segment_6'],
                'historical_speed': 30.0
            },
            'segment_5': {
                'geometry': {'x': 30.905000, 'y': 75.845000},
                'length': 700,
                'connects_to': ['segment_6'],
                'historical_speed': 50.0
            },
            'segment_6': {
                'geometry': {'x': 30.900000, 'y': 75.840000},
                'length': 900,
                'connects_to': [],
                'historical_speed': 55.0
            }
        }
        # Make connectivity bidirectional
        for seg, data in graph.items():
            for neighbor in data['connects_to']:
                if neighbor in graph and seg not in graph[neighbor]['connects_to']:
                    graph[neighbor]['connects_to'].append(seg)
        return graph

    def _nearest_segment(self, coords):
        """Find nearest road segment to given coordinates"""
        min_dist = float('inf')
        nearest_segment = None

        for segment_id, data in self.route_graph.items():
            dx = coords[0] - data['geometry']['x']
            dy = coords[1] - data['geometry']['y']
            distance = math.sqrt(dx**2 + dy**2)

            if distance < min_dist:
                min_dist = distance
                nearest_segment = segment_id

        return nearest_segment

    def _get_features(self, segment_id, time_obj):
        """Prepare input features for model prediction"""
        segment_data = self.route_graph.get(segment_id, {})
        if not segment_data:
            return None

        # Ensure time_obj is a datetime
        if not isinstance(time_obj, datetime):
            raise ValueError("departure_time must be a datetime.datetime object")

        return np.array([
            segment_data['length'],
            segment_data['historical_speed'],
            time_obj.hour,  # Time-based features
            time_obj.weekday(),
            # Add more features as needed
        ], dtype=np.float32).reshape(1, -1)

    def predict_segment(self, segment_id, features):
        features = np.array(features, dtype=np.float32).reshape(1, -1, 1)
        print("features shape:", features.shape)
        print("features:", features)
        
        model_inputs = self.feature_generator.get_model_input_features(segment_id)

        if model_inputs is None:
            print(f"[Warning] Insufficient features for segment {segment_id}. Falling back to historical speed.")
            # Fallback to historical speed prediction
            historical_speed = self.route_graph[segment_id]['historical_speed']
            return historical_speed  # directly return speed, not model output

        temporal_features = model_inputs['temporal']  # shape (1,10,5)
        spatial_features = model_inputs['spatial']    # shape (1,3)

        # Expand dims to batch size = 1
        temporal_features = np.expand_dims(temporal_features, axis=0)
        spatial_features = np.expand_dims(spatial_features, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], temporal_features)
        self.interpreter.set_tensor(self.input_details[1]['index'], spatial_features)
        
        # Invoke the interpreter to actually run the inference
        self.interpreter.invoke()
        
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data[0][0]

    def _estimate_travel_time(self, segment_id, departure_time):
        """Calculate estimated travel time and speed for segment"""
        # Convert numpy.datetime64 to python datetime
        if isinstance(departure_time, np.datetime64):
            departure_time = departure_time.astype('datetime64[s]').tolist()
    
        features = self._get_features(segment_id, departure_time)
        if features is None:
            # Fallback to historical speed
            segment_data = self.route_graph.get(segment_id)
            if segment_data:
                speed = segment_data.get('historical_speed', 30.0)
                travel_time_seconds = segment_data['length'] / (speed * 1000 / 3600)
                return timedelta(seconds=travel_time_seconds), speed  # Return time and historical speed
            else:
                return timedelta(seconds=0), 0.0
    
        # Get predicted speed (or congestion level)
        prediction = self.predict_segment(segment_id, features)
        
        # Check if the prediction is a float (scalar value)
        if isinstance(prediction, float):
            predicted_speed = prediction  # directly use prediction if it's a scalar
        else:
            predicted_speed = float(prediction[0])  # else extract the value from array-like structure
    
        segment_data = self.route_graph[segment_id]
        distance = segment_data['length']  # meters
    
        # Basic calculation: time = distance / speed
        if predicted_speed <= 0:
            predicted_speed = 5  # Minimum speed to avoid division by zero
    
        # Convert speed from km/h to m/s: km/h * 1000/3600
        speed_m_s = predicted_speed * 1000 / 3600
        travel_time_seconds = distance / speed_m_s
    
        return timedelta(seconds=travel_time_seconds), predicted_speed


    def find_optimal_route(self, start_coords, end_coords, departure_time):
        """A* algorithm implementation for route finding"""
        # Determine nearest segments
        start_segment = self._nearest_segment(start_coords)
        end_segment = self._nearest_segment(end_coords)

        if not start_segment or not end_segment:
            return {"error": "Could not locate start/end segments"}

        # Convert departure_time
        if isinstance(departure_time, np.datetime64):
            departure_time = departure_time.astype('datetime64[s]').tolist()
        elif not isinstance(departure_time, datetime):
            departure_time = datetime.now()

        # Heuristic function (Euclidean distance)
        def heuristic(segment_id):
            end_x = self.route_graph[end_segment]['geometry']['x']
            end_y = self.route_graph[end_segment]['geometry']['y']
            seg_x = self.route_graph[segment_id]['geometry']['x']
            seg_y = self.route_graph[segment_id]['geometry']['y']
            return math.sqrt((end_x - seg_x)**2 + (end_y - seg_y)**2)

        # Priority queue: (f_score, g_score, current_path)
        open_set = PriorityQueue()
        open_set.put((0, 0, [start_segment]))

        g_scores = defaultdict(lambda: float('inf'))
        g_scores[start_segment] = 0

        visited = set()

        while not open_set.empty():
            _, g_score, path = open_set.get()
            current = path[-1]

            # Check goal
            if current == end_segment:
                return {
                    "segments": path,
                    "total_time": str(timedelta(seconds=g_score)),
                    "total_time_seconds": g_score,
                    "waypoints": [
                        self.route_graph[seg]['geometry'] for seg in path
                    ]
                }

            if current in visited:
                continue
            visited.add(current)

            # Explore neighbors (bidirectional)
            neighbors = set(self.route_graph[current].get('connects_to', []))
            for seg_id, data in self.route_graph.items():
                if current in data.get('connects_to', []):
                    neighbors.add(seg_id)

            for neighbor in neighbors:
                # Calculate time to neighbor
                time_delta, predicted_speed = self._estimate_travel_time(neighbor, departure_time)
                time_seconds = time_delta.total_seconds()

                tentative_g = g_score + time_seconds

                if tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    open_set.put((f_score, tentative_g, path + [neighbor]))

        return {"error": "No valid route found"}

    def get_route_details(self, segments):
        """Get detailed information for route segments including estimated speed"""
        now = datetime.now()
        route_details = []
        for seg_id in segments:
            estimated_time, estimated_speed = self._estimate_travel_time(seg_id, now)
            route_details.append({
                "id": seg_id,
                "position": self.route_graph[seg_id]['geometry'],
                "length": self.route_graph[seg_id]['length'],
                "estimated_time": str(estimated_time),
                "estimated_speed": estimated_speed  # Add estimated speed
            })
        return route_details
