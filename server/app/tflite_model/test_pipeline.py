# tflite_model/test_pipeline.py
import os
import sys
import numpy as np
import tensorflow as tf
from django.apps import apps
import django

# Configure Django Environment
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
django.setup()

# Now import Django-dependent modules
from ml_pipeline.data_processing import RealTimeDataProcessor
from ml_pipeline.feature_engineering import StreamFeatureGenerator
from ml_pipeline.convert_to_tflite import convert_to_tflite
from ml_pipeline.route_optimizer import RouteOptimizer

def test_pipeline():
    print("Starting Real-Time Pipeline Testing...")
    
    # Initialize components
    processor = RealTimeDataProcessor()
    feature_gen = StreamFeatureGenerator()
    TrafficData = apps.get_model('app', 'TrafficData')
    
    # Test processing
    test_data = TrafficData.objects.last()
    processed = processor.process_record(test_data)
    assert 'normalized_speed' in processed, "Data processing failed"
    
    # Test features
    feature_window = feature_gen.update_features([processed])
    assert len(feature_window) > 0, "Feature generation failed"
    
    # Test conversion
    convert_to_tflite(
        "models/traffic_lstm_model.h5",
        "models/traffic_lstm_model.tflite"
    )
    assert os.path.exists("models/traffic_lstm_model.tflite"), "Conversion failed"
    
    # Test optimization
    optimizer = RouteOptimizer(tflite_model="models/traffic_lstm_model.tflite")
    test_route = optimizer.find_optimal_route(
        start_coords=(30.816106, 75.867321),
        end_coords=(30.7337, 76.7794),
        departure_time=np.datetime64('now')
    )
    assert 'segments' in test_route, "Optimization failed"
    
    print("All Tests Passed!")

if __name__ == "__main__":
    test_pipeline()
