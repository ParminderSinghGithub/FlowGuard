from django.db import models
from django.contrib.auth.models import AbstractUser

# Create your models here.

class User(AbstractUser):
    device_id = models.CharField(max_length=255, unique=True)  # Track device anonymously
    is_active_user = models.BooleanField(default=True)  # Track if currently active in-app
    preferred_routes = models.ManyToManyField('Route', blank=True)  # User-saved routes

class TrafficData(models.Model):
    location = models.CharField(max_length=255)
    latitude = models.FloatField()
    longitude = models.FloatField()
    congestion_level = models.CharField(max_length=50)  # E.g., 'low', 'moderate', 'high'
    timestamp = models.DateTimeField(auto_now_add=True)  # Record data time

class CongestionPrediction(models.Model):
    location = models.ForeignKey(TrafficData, on_delete=models.CASCADE)
    predicted_congestion_level = models.CharField(max_length=50)
    prediction_time = models.DateTimeField()
    accuracy = models.DecimalField(max_digits=4, decimal_places=2)  # Prediction accuracy score

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
