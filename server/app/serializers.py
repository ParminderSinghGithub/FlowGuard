from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from django.utils import timezone
from datetime import timedelta
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


class SignupSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True, trim_whitespace=False)
    password_confirmation = serializers.CharField(
        write_only=True,
        trim_whitespace=False,
        required=True,
        allow_blank=False,
    )

    def validate_username(self, value):
        normalized = value.strip()

        if not normalized:
            raise serializers.ValidationError('Username cannot be empty.')

        if User.objects.filter(username__iexact=normalized).exists():
            raise serializers.ValidationError(
                'A user with this username already exists.'
            )

        return normalized

    def validate(self, attrs):
        password = attrs.get('password')
        password_confirmation = attrs.get('password_confirmation')

        if password != password_confirmation:
            raise serializers.ValidationError({
                'password_confirmation': 'Passwords do not match.'
            })

        user = User(username=attrs.get('username'))
        validate_password(password, user=user)

        return attrs

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
    timestamp = serializers.DateTimeField(required=False)

    def validate_latitude(self, value):
        if value < -90.0 or value > 90.0:
            raise serializers.ValidationError('Latitude must be between -90 and 90.')
        return value

    def validate_longitude(self, value):
        if value < -180.0 or value > 180.0:
            raise serializers.ValidationError('Longitude must be between -180 and 180.')
        return value

    def validate_accelerometer_z(self, value):
        if value < -50.0 or value > 50.0:
            raise serializers.ValidationError('Accelerometer Z-axis value out of allowed range (-50 to 50).')
        return value

    def validate_timestamp(self, value):
        now = timezone.now()
        if value > now + timedelta(minutes=5):
            raise serializers.ValidationError('Timestamp cannot be in the far future.')
        if value < now - timedelta(days=2):
            raise serializers.ValidationError('Timestamp is too old for live ingestion.')
        return value


class NearbyPotholeQuerySerializer(serializers.Serializer):
    latitude = serializers.FloatField()
    longitude = serializers.FloatField()
    radius_meters = serializers.FloatField(required=False, default=250.0, min_value=10.0, max_value=5000.0)
    verified_only = serializers.BooleanField(required=False, default=True)

    def validate_latitude(self, value):
        if value < -90.0 or value > 90.0:
            raise serializers.ValidationError('Latitude must be between -90 and 90.')
        return value

    def validate_longitude(self, value):
        if value < -180.0 or value > 180.0:
            raise serializers.ValidationError('Longitude must be between -180 and 180.')
        return value


class RouteWarningQuerySerializer(serializers.Serializer):
    latitude = serializers.FloatField()
    longitude = serializers.FloatField()
    radius_meters = serializers.FloatField(required=False, default=500.0, min_value=25.0, max_value=10000.0)

    def validate_latitude(self, value):
        if value < -90.0 or value > 90.0:
            raise serializers.ValidationError('Latitude must be between -90 and 90.')
        return value

    def validate_longitude(self, value):
        if value < -180.0 or value > 180.0:
            raise serializers.ValidationError('Longitude must be between -180 and 180.')
        return value


class RouteOptimizationRequestSerializer(serializers.Serializer):
    start_latitude = serializers.FloatField()
    start_longitude = serializers.FloatField()
    end_latitude = serializers.FloatField()
    end_longitude = serializers.FloatField()
    departure_time = serializers.DateTimeField(required=False)
    eta_tolerance_ratio = serializers.FloatField(required=False, default=1.15, min_value=1.0, max_value=2.0)
    alternatives_count = serializers.IntegerField(required=False, default=3, min_value=1, max_value=6)

    def validate(self, attrs):
        for key in ('start_latitude', 'end_latitude'):
            if attrs[key] < -90.0 or attrs[key] > 90.0:
                raise serializers.ValidationError({key: 'Latitude must be between -90 and 90.'})

        for key in ('start_longitude', 'end_longitude'):
            if attrs[key] < -180.0 or attrs[key] > 180.0:
                raise serializers.ValidationError({key: 'Longitude must be between -180 and 180.'})

        if (
            attrs['start_latitude'] == attrs['end_latitude']
            and attrs['start_longitude'] == attrs['end_longitude']
        ):
            raise serializers.ValidationError('Start and end coordinates cannot be identical.')

        if attrs.get('departure_time') and attrs['departure_time'] < timezone.now() - timedelta(hours=12):
            raise serializers.ValidationError({'departure_time': 'Departure time is too far in the past.'})

        return attrs


class RouteRiskAnalysisSerializer(serializers.Serializer):
    start_latitude = serializers.FloatField()
    start_longitude = serializers.FloatField()
    end_latitude = serializers.FloatField()
    end_longitude = serializers.FloatField()

    def validate(self, attrs):
        for key in ('start_latitude', 'end_latitude'):
            if attrs[key] < -90.0 or attrs[key] > 90.0:
                raise serializers.ValidationError({key: 'Latitude must be between -90 and 90.'})

        for key in ('start_longitude', 'end_longitude'):
            if attrs[key] < -180.0 or attrs[key] > 180.0:
                raise serializers.ValidationError({key: 'Longitude must be between -180 and 180.'})

        return attrs


class UserLocationSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserLocation
        fields = ['user', 'latitude', 'longitude', 'updated_at']

