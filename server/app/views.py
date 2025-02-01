import json
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
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

# Load the TFLite model
TFLITE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'tflite_model/traffic_lstm_model.tflite')
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@csrf_exempt
def predict_traffic(request):
    """
    API endpoint to predict traffic. Expects JSON input with original features:
    'hour', 'day_of_week', 'is_weekend'.
    """
    if request.method == 'POST':
        try:
            # Parse JSON body
            body = json.loads(request.body.decode('utf-8'))

            # Step 1: Retrieve historical vehicle data or simulate data
            recent_data = list(TrafficData.objects.order_by('-timestamp')[:6])

            if len(recent_data) < 6:
                # Simulate dynamic data based on input hour for variety
                hour = body['hour']
                simulated_vehicles = [10 + hour % 5, 15 + hour % 7, 20 + hour % 3, 25, 18, 12]
                recent_data = [{'vehicles': v} for v in simulated_vehicles[:6]]

            # Create DataFrame for feature engineering
            df = pd.DataFrame({
                'Hour': [body['hour']] * len(recent_data),
                'DayOfWeek': [body['day_of_week']] * len(recent_data),
                'IsWeekend': [body['is_weekend']] * len(recent_data),
                'Vehicles': [data['vehicles'] if isinstance(data, dict) else data.vehicles for data in recent_data]
            })

            # Step 2: Normalize 'Vehicles' using a pre-trained MinMaxScaler
            scaler = joblib.load('app/tflite_model/scaler.pkl')  # Load pre-trained scaler
            df['Vehicles_Normalized'] = scaler.transform(df[['Vehicles']])

            # Step 3: Add lag and rolling mean features
            df['Lag_1'] = df['Vehicles_Normalized'].shift(1)
            df['Lag_2'] = df['Vehicles_Normalized'].shift(2)
            df['Lag_3'] = df['Vehicles_Normalized'].shift(3)
            df['Rolling_Mean_3'] = df['Vehicles_Normalized'].rolling(window=3).mean()

            # Drop NaN rows caused by shifting
            df.dropna(inplace=True)

            # Ensure sufficient rows for input preparation
            if len(df) < 3:
                return JsonResponse({'error': 'Not enough data for feature engineering.'}, status=400)

            # Step 4: Prepare input for the model (last 3 timesteps)
            input_array = df[['Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean_3']].tail(3).values
            input_array = input_array.astype(np.float32)  # Convert to float32
            input_array = np.expand_dims(input_array, axis=0)  # Shape: (1, time_steps, features)

            # Step 5: Perform prediction
            interpreter.set_tensor(input_details[0]['index'], input_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = output_data[0][0]

            # Step 6: Denormalize the prediction
            denormalized_prediction = scaler.inverse_transform(np.array([[prediction]]))[0][0]

            # Step 7: Return response
            return JsonResponse({'prediction': float(denormalized_prediction)})

        except KeyError as e:
            return JsonResponse({'error': f'Missing parameter: {e.args[0]}'}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request method. Use POST.'}, status=405)


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
