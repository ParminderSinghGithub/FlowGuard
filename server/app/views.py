import json
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from sklearn.preprocessing import MinMaxScaler
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from .models import User, TrafficData, CongestionPrediction, PotholeReport, Notification, Route
from .serializers import (
    UserSerializer, TrafficDataSerializer, CongestionPredictionSerializer,
    PotholeReportSerializer, NotificationSerializer, RouteSerializer
)
from .traffic_apis.tomtom import get_ludhiana_traffic  # Ludhiana-specific API

# Load TFLite model (Ludhiana-trained)
TFLITE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'tflite_model/models/traffic_lstm_model.tflite')
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load MinMaxScaler
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'tflite_model/scaler.pkl')
scaler = joblib.load(SCALER_PATH)

# Ludhiana-specific constants
LUDHIANA_HOTSPOTS = [
    (30.9000, 75.8573),  # City Center
    (30.9158, 75.8227),  # PAU/Sarabha Nagar
    (30.8412, 75.8573),  # Bus Stand
    (30.8786, 75.8000)   # Dugri Rd
]

@csrf_exempt
def predict_traffic(request):
    """Predict congestion for Ludhiana hotspots using TFLite model."""
    if request.method == 'POST':
        try:
            # Get real-time data from TomTom (Ludhiana)
            traffic_data = get_ludhiana_traffic()
            if not traffic_data:
                return JsonResponse({"error": "Failed to fetch Ludhiana traffic data"}, status=500)

            # Prepare input for TFLite model
            input_array = []
            for segment in traffic_data:
                # Use normalized speed ratio as feature
                speed_ratio = segment['speeds']['current'] / segment['speeds']['free_flow']
                input_array.append([speed_ratio])
            
            input_array = np.array(input_array[-3:], dtype=np.float32)  # Last 3 timesteps
            input_array = np.expand_dims(input_array, axis=0)  # Shape: (1, 3, 1)

            # Predict
            interpreter.set_tensor(input_details[0]['index'], input_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
            denormalized_pred = scaler.inverse_transform([[prediction]])[0][0]

            # Save prediction
            latest_data = TrafficData.objects.latest('timestamp')
            CongestionPrediction.objects.create(
                location=latest_data,
                predicted_congestion_level='severe' if denormalized_pred < 0.4 else 'moderate',
                prediction_time=timezone.now(),
                accuracy=0.95  # Placeholder
            )

            return JsonResponse({
                "prediction": float(denormalized_pred),
                "hotspots": [{"lat": lat, "lon": lon} for lat, lon in LUDHIANA_HOTSPOTS]
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "POST method required"}, status=405)

def test_traffic_flow(request):
    """Test endpoint for Ludhiana traffic (TomTom)."""
    flow_data = get_ludhiana_traffic()
    return JsonResponse(flow_data, safe=False) if flow_data else \
           JsonResponse({"error": "API failure"}, status=500)

# ViewSets (unchanged, but ensure they use Ludhiana data filters)
class TrafficDataViewSet(viewsets.ModelViewSet):
    queryset = TrafficData.objects.filter(location__icontains="Ludhiana")  # Ludhiana-only
    serializer_class = TrafficDataSerializer

class CongestionPredictionViewSet(viewsets.ModelViewSet):
    queryset = CongestionPrediction.objects.filter(location__location__icontains="Ludhiana")
    serializer_class = CongestionPredictionSerializer

# Create your views here.
def home(request):
    return HttpResponse("Welcome to FlowGuard App")

# User ViewSet
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

    # Example: retrieve user's routes
    @action(detail=True, methods=['get'])
    def routes(self, request, pk=None):
        user = self.get_object()
        routes = user.preferred_routes.all()
        serializer = RouteSerializer(routes, many=True)
        return Response(serializer.data)

# TrafficData ViewSet
class TrafficDataViewSet(viewsets.ModelViewSet):
    queryset = TrafficData.objects.all()
    serializer_class = TrafficDataSerializer

    # Retrieve traffic data for a specific location
    @action(detail=False, methods=['get'])
    def location_data(self, request):
        latitude = request.query_params.get('latitude')
        longitude = request.query_params.get('longitude')
        if latitude and longitude:
            traffic_data = TrafficData.objects.filter(latitude=latitude, longitude=longitude)
            serializer = self.get_serializer(traffic_data, many=True)
            return Response(serializer.data)
        return Response({"error": "Location parameters missing."}, status=status.HTTP_400_BAD_REQUEST)

# CongestionPrediction ViewSet
class CongestionPredictionViewSet(viewsets.ModelViewSet):
    queryset = CongestionPrediction.objects.all()
    serializer_class = CongestionPredictionSerializer

    # Get predictions for a specific location
    @action(detail=False, methods=['get'])
    def location_prediction(self, request):
        location_id = request.query_params.get('location_id')
        if location_id:
            predictions = CongestionPrediction.objects.filter(location_id=location_id)
            serializer = self.get_serializer(predictions, many=True)
            return Response(serializer.data)
        return Response({"error": "Location ID missing."}, status=status.HTTP_400_BAD_REQUEST)

# PotholeReport ViewSet
class PotholeReportViewSet(viewsets.ModelViewSet):
    queryset = PotholeReport.objects.all()
    serializer_class = PotholeReportSerializer

    # Add a new pothole report (POST)
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    # Get verified potholes only
    @action(detail=False, methods=['get'])
    def verified_potholes(self, request):
        verified_potholes = PotholeReport.objects.filter(is_verified=True)
        serializer = self.get_serializer(verified_potholes, many=True)
        return Response(serializer.data)

# Notification ViewSet
class NotificationViewSet(viewsets.ModelViewSet):
    queryset = Notification.objects.all()
    serializer_class = NotificationSerializer

    # Mark notification as read
    @action(detail=True, methods=['post'])
    def mark_as_read(self, request, pk=None):
        notification = self.get_object()
        notification.is_read = True
        notification.save()
        return Response({"status": "Notification marked as read."})

# Route ViewSet
class RouteViewSet(viewsets.ModelViewSet):
    queryset = Route.objects.all()
    serializer_class = RouteSerializer

    # Get traffic data for a specific route
    @action(detail=True, methods=['get'])
    def traffic_data(self, request, pk=None):
        route = self.get_object()
        traffic_data = route.traffic_data.all()
        serializer = TrafficDataSerializer(traffic_data, many=True)
        return Response(serializer.data)
