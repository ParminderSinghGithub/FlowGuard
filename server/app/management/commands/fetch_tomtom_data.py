# app/management/commands/fetch_tomtom_data.py
import logging

logger = logging.getLogger(__name__)

def fetch_and_save_data():
    """Fetch and process Ludhiana traffic; return summary string."""
    import os
    import django
    from django.utils import timezone
    from django.db import transaction
    from app.models import TrafficData
    from app.traffic_apis.tomtom import get_ludhiana_traffic, LUDHIANA_HOTSPOTS

    # Set up Django environment
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
    django.setup()

    logger.info("Starting fetch_and_save_data()")
    try:
        data = get_ludhiana_traffic()
        total_hotspots = len(LUDHIANA_HOTSPOTS)
        if not data:
            logger.warning("No traffic data returned")
            return "No data"

        saved_count = 0
        with transaction.atomic():
            for i, segment in enumerate(data):
                lat, lon, _ = LUDHIANA_HOTSPOTS[i]
                try:
                    logger.info(f"[{i+1}/{total_hotspots}] Saving segment for {lat},{lon}")
                    traffic_obj = save_road_segment(segment)
                    saved_count += 1
                except Exception as seg_err:
                    logger.error(f"Failed to save segment {i+1}: {seg_err}")

        logger.info(f"Processed {saved_count}/{total_hotspots} segments")
        return f"Saved {saved_count} of {total_hotspots}"

    except Exception as e:
        logger.exception("Ludhiana data error in fetch_and_save_data()")
        raise

def save_road_segment(segment):
    try:
        from django.utils import timezone
        from app.models import TrafficData
        timestamp = timezone.now().replace(second=0, microsecond=0)
        coordinates = segment['coordinates']['coordinate'][0]
        latitude = round(float(coordinates['latitude']), 6)
        longitude = round(float(coordinates['longitude']), 6)
        current_speed = float(segment['speeds'].get('current', 20.0))
        free_flow_speed = float(segment['speeds'].get('free_flow', 40.0))
        
        obj, created = TrafficData.objects.update_or_create(
            latitude=latitude,
            longitude=longitude,
            timestamp=timestamp,
            defaults={
                'latitude': latitude,
                'longitude': longitude,
                'road_type': segment.get('road_type', 'UNKNOWN'),
                'current_speed': current_speed,
                'free_flow_speed': free_flow_speed,
                'congestion_level': segment.get('congestion_level', 'unknown'),
                'is_bottleneck': segment.get('is_bottleneck', False)
            }
        )
        return obj
    except Exception as e:
        logger.error(f"Save error: {str(e)}")
        raise

def calculate_congestion(current, free_flow):
    """Ludhiana-specific thresholds"""
    ratio = current / free_flow
    if ratio < 0.4: return 'severe'
    elif ratio < 0.7: return 'moderate'
    return 'light'

def save_traffic_flow(data):
    """Save traffic flow data to TrafficData model."""
    from django.utils import timezone
    from app.models import TrafficData

    try:
        # Validate required fields
        required_fields = ['currentSpeed', 'freeFlowSpeed', 'coordinates']
        if not all(field in data for field in required_fields):
            logger.error("Missing required fields in traffic data")
            return

        # Safely extract coordinates
        coordinates = data['coordinates']['coordinate'][0] if data['coordinates']['coordinate'] else None
        if not coordinates:
            logger.error("No coordinates in traffic data")
            return

        # Handle division safely
        try:
            current_speed = float(data['currentSpeed'])
            free_flow_speed = float(data['freeFlowSpeed'])

            if free_flow_speed <= 0:
                logger.warning(f"Invalid freeFlowSpeed: {free_flow_speed}")
                congestion_level = 'high'
            else:
                congestion_level = map_congestion(current_speed / free_flow_speed)

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid speed values: {str(e)}")
            congestion_level = 'unknown'

        TrafficData.objects.create(
            location="TomTom Flow Data",
            latitude=coordinates['latitude'],
            longitude=coordinates['longitude'],
            congestion_level=congestion_level,
            timestamp=timezone.now()
        )
        logger.info("Saved traffic flow data")

    except Exception as e:
        logger.error(f"Error saving traffic data: {str(e)}")

def map_congestion(speed_ratio):
    """Convert TomTom speed ratio to your congestion levels (low/moderate/high)."""
    try:
        if speed_ratio < 0.5:
            return 'high'
        elif 0.5 <= speed_ratio < 0.8:
            return 'moderate'
        return 'low'
    except TypeError:
        return 'unknown'

from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Fetches real-time traffic data from TomTom API and saves to DB"

    def handle(self, *args, **options):
        fetch_and_save_data()
