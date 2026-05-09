from rest_framework import serializers
from .models import (
    User,
    TrafficData,
    CongestionPrediction,
    PotholeReport,
    Notification,
    Route,
    PotholeCluster,
    SensorDataPoint,
    UserLocation,
)

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
        fields = [
            'id',
            'user',
            'source_device_id',
            'latitude',
            'longitude',
            'severity',
            'source_type',
            'accelerometer_z',
            'confidence_score',
            'timestamp',
            'is_verified',
            'cluster',
        ]

class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = ['id', 'user', 'message', 'timestamp', 'is_read']

class RouteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Route
        fields = ['id', 'start_point', 'end_point', 'traffic_data']


class SensorDataPointSerializer(serializers.ModelSerializer):
    class Meta:
        model = SensorDataPoint
        fields = ['id', 'user', 'device_id', 'latitude', 'longitude', 'accelerometer_z', 'recorded_at']


class PotholeClusterSerializer(serializers.ModelSerializer):
    class Meta:
        model = PotholeCluster
        fields = [
            'id',
            'centroid_latitude',
            'centroid_longitude',
            'radius_meters',
            'reports_count',
            'confidence_aggregate',
            'is_verified',
            'last_verified_at',
            'created_at',
            'updated_at',
        ]


class SensorIngestSerializer(serializers.Serializer):
    device_id = serializers.CharField(max_length=255)
    user_id = serializers.IntegerField(required=False)
    latitude = serializers.FloatField()
    longitude = serializers.FloatField()
    accelerometer_z = serializers.FloatField()


class NearbyPotholeQuerySerializer(serializers.Serializer):
    latitude = serializers.FloatField()
    longitude = serializers.FloatField()
    radius_meters = serializers.FloatField(required=False, default=250.0, min_value=10.0, max_value=5000.0)
    verified_only = serializers.BooleanField(required=False, default=True)


class RouteWarningQuerySerializer(serializers.Serializer):
    latitude = serializers.FloatField()
    longitude = serializers.FloatField()
    radius_meters = serializers.FloatField(required=False, default=500.0, min_value=25.0, max_value=10000.0)


class UserLocationSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserLocation
        fields = ['user', 'latitude', 'longitude', 'updated_at']

