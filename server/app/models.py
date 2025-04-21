from django.db import models
from django.utils.crypto import get_random_string  
from django.contrib.auth.models import AbstractUser
import uuid

# Create your models here.

class User(AbstractUser):
    device_id = models.CharField(max_length=255, unique=True)  # Track device anonymously
    is_active_user = models.BooleanField(default=True)  # Track if currently active in-app
    preferred_routes = models.ManyToManyField('Route', blank=True)  # User-saved routes

# models.py
class TrafficData(models.Model):
    ROAD_CLASS_CHOICES = [
        ('FRC0', 'Highway (NH5)'),
        ('FRC1', 'Major Road (Ferozepur Rd)'),
        ('FRC2', 'Secondary Road'),
        ('FRC3', 'Local Road'),
        ('FRC4', 'Ramp/Exit Road'),
        ('FRC5', 'Alley/Private Road'),
        ('FRC6', 'Other'),
        ('UNKNOWN', 'Unknown')
    ]
    
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        unique=True
    )
    location = models.CharField(max_length=255, default='Ludhiana')
    latitude = models.DecimalField(max_digits=9, decimal_places=6)
    longitude = models.DecimalField(max_digits=9, decimal_places=6)
    road_type = models.CharField(
        max_length=10, 
        choices=ROAD_CLASS_CHOICES, 
        default='UNKNOWN'
    )
    current_speed = models.FloatField(default=20.0)  # Reasonable default speed in km/h
    free_flow_speed = models.FloatField(default=40.0)  # Typical free flow speed
    congestion_level = models.CharField(
        max_length=20, 
        default='unknown',
        choices=[
            ('unknown', 'Unknown'),
            ('light', 'Light'),
            ('moderate', 'Moderate'),
            ('severe', 'Severe')
        ]
    )
    timestamp = models.DateTimeField(auto_now_add=True)
    is_bottleneck = models.BooleanField(default=False)
    confidence_score = models.FloatField(default=0.9)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['latitude', 'longitude', 'timestamp'],
                name='unique_location_time'
            )
    ]
        indexes = [
        models.Index(fields=['timestamp', 'latitude', 'longitude']),
    ]

    def __str__(self):
        return f"{self.location} @ {self.timestamp}"

class CongestionPrediction(models.Model):
    location = models.ForeignKey(TrafficData, on_delete=models.CASCADE)
    predicted_congestion_level = models.CharField(max_length=50)
    prediction_time = models.DateTimeField()
    accuracy = models.DecimalField(max_digits=4, decimal_places=2)  # Prediction accuracy score
    prediction_confidence = models.FloatField(default=0.7)  # Added default
    actual_congestion = models.CharField(max_length=20, null=True, blank=True)  # Made nullable
    error_margin = models.FloatField(null=True, blank=True)  # Made nullable
    is_bottleneck_affected = models.BooleanField(default=False)  # Added default

class PotholeReport(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    latitude = models.FloatField()
    longitude = models.FloatField()
    severity = models.CharField(max_length=20)  # E.g., 'minor', 'moderate', 'severe'
    timestamp = models.DateTimeField(auto_now_add=True)
    is_verified = models.BooleanField(default=False)  # Track if verified by repeated detection

class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

class Route(models.Model):
    start_point = models.CharField(max_length=255)
    end_point = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    traffic_data = models.ManyToManyField(TrafficData)
