from celery import shared_task
import logging
from django.db import connection
from django.utils import timezone
from datetime import timedelta

from app.models import SensorDataPoint, Notification, PotholeCluster
from app.services.pothole_service import PotholeService

logger = logging.getLogger(__name__)

@shared_task(bind=True, 
             name="app.tasks.fetch_ludhiana_traffic",  # Simplified name
             max_retries=3,
             autoretry_for=(Exception,),
             retry_backoff=True)
def fetch_ludhiana_traffic(self):
    """Fetch traffic data with enhanced error handling"""
    try:
        import time
        start = time.time()
        logger.info("Task started - refreshing DB connection")
        connection.close()
        
        from app.management.commands.fetch_tomtom_data import fetch_and_save_data
        result = fetch_and_save_data()
        
        logger.info(f"TASK COMPLETED in {time.time()-start:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Task failed: {str(e)}")
        raise self.retry(exc=e)


@shared_task(name='app.tasks.verify_pothole_clusters_task')
def verify_pothole_clusters_task():
    service = PotholeService()
    return service.verify_all_clusters()


@shared_task(name='app.tasks.cleanup_stale_sensor_points_task')
def cleanup_stale_sensor_points_task(hours=24):
    cutoff = timezone.now() - timedelta(hours=hours)
    deleted, _ = SensorDataPoint.objects.filter(recorded_at__lt=cutoff).delete()
    return {'deleted_sensor_points': deleted}


@shared_task(name='app.tasks.cleanup_stale_notifications_task')
def cleanup_stale_notifications_task(days=14):
    cutoff = timezone.now() - timedelta(days=days)
    deleted, _ = Notification.objects.filter(timestamp__lt=cutoff, is_read=True).delete()
    return {'deleted_notifications': deleted}


@shared_task(name='app.tasks.cleanup_inactive_clusters_task')
def cleanup_inactive_clusters_task(days=30):
    cutoff = timezone.now() - timedelta(days=days)
    stale_clusters = PotholeCluster.objects.filter(updated_at__lt=cutoff, is_verified=False)
    deleted, _ = stale_clusters.delete()
    return {'deleted_clusters': deleted}


@shared_task(name='app.tasks.refresh_route_cache_task')
def refresh_route_cache_task():
    # Placeholder for future cache backend integration; safe no-op now.
    return {'refreshed': True, 'cache_backend': 'none'}