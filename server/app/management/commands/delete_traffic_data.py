# app/management/commands/delete_traffic_data.py
from django.core.management.base import BaseCommand
from app.models import TrafficData
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Deletes all TrafficData entries from the database"

    def handle(self, *args, **options):
        try:
            count = TrafficData.objects.all().count()
            TrafficData.objects.all().delete()
            logger.info(f"Deleted {count} TrafficData entries.")
        except Exception as e:
            logger.error(f"Error deleting TrafficData: {str(e)}")