from rest_framework import serializers
from .models import User, TrafficData, CongestionPrediction, PotholeReport, Notification, Route

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'device_id', 'is_active_user']

class TrafficDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrafficData
        fields = ['id', 'location', 'latitude', 'longitude', 'congestion_level', 'timestamp']

class CongestionPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = CongestionPrediction
        fields = ['id', 'location', 'predicted_congestion_level', 'prediction_time', 'accuracy']

class PotholeReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = PotholeReport
        fields = ['id', 'user', 'latitude', 'longitude', 'severity', 'timestamp', 'is_verified']

class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = ['id', 'user', 'message', 'timestamp', 'is_read']

class RouteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Route
        fields = ['id', 'start_point', 'end_point', 'traffic_data']
