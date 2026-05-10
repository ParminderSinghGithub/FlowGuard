"""
Migration 0004: Add spatial and query performance indexes.

Addresses Tier 1 audit findings:
- Missing indexes on geographic fields causing full table scans
- Missing compound indexes on frequently-queried field patterns
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_potholecluster_potholereport_accelerometer_z_and_more'),
    ]

    operations = [
        # Geographic query indexes on PotholeReport (nearby_potholes queries)
        migrations.AddIndex(
            model_name='potholereport',
            index=models.Index(fields=['latitude', 'longitude'], name='pothole_geo_idx'),
        ),
        
        # Device-based queries with timestamp
        migrations.AddIndex(
            model_name='potholereport',
            index=models.Index(fields=['source_device_id', 'timestamp'], name='pothole_device_time_idx'),
        ),
        
        # Cluster centroid geographic queries (route_service)
        migrations.AddIndex(
            model_name='potholecluster',
            index=models.Index(fields=['centroid_latitude', 'centroid_longitude'], name='cluster_geo_idx'),
        ),
        
        # Verified cluster status queries (common in route scoring)
        migrations.AddIndex(
            model_name='potholecluster',
            index=models.Index(fields=['is_verified', 'updated_at'], name='cluster_verified_time_idx'),
        ),
        
        # Notification cleanup queries (stale read notifications)
        migrations.AddIndex(
            model_name='notification',
            index=models.Index(fields=['is_read', 'timestamp'], name='notification_read_time_idx'),
        ),
        
        # UserLocation geographic queries (nearby user notifications)
        migrations.AddIndex(
            model_name='userlocation',
            index=models.Index(fields=['latitude', 'longitude', 'user_id'], name='user_location_geo_idx'),
        ),
        
        # SensorDataPoint baseline queries (device_id + recent readings)
        migrations.AddIndex(
            model_name='sensordatapoint',
            index=models.Index(fields=['device_id', 'recorded_at'], name='sensor_device_time_idx'),
        ),
    ]
