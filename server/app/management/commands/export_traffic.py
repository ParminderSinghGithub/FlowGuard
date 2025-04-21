from django.core.management.base import BaseCommand
import pandas as pd
import os
from django.utils import timezone
from datetime import timedelta
from app.models import TrafficData

class Command(BaseCommand):
    help = 'Exports traffic data to CSV'

    def handle(self, *args, **options):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_path = os.path.join(project_root, 'tflite_model', 'traffic_data.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Debug: Print some model info
        total_records = TrafficData.objects.count()
        self.stdout.write(f"Total records in DB: {total_records}")

        # Get data with proper time filtering
        qs = TrafficData.objects.filter(
            timestamp__gte=timezone.now()-timedelta(days=1)
        ).values('timestamp', 'latitude', 'longitude', 'current_speed', 'free_flow_speed')
        
        self.stdout.write(f"Found {qs.count()} records in last 24 hours")
        
        df = pd.DataFrame.from_records(qs)  # Changed from DataFrame()
        
        if not df.empty:
            df.to_csv(output_path, index=False)
            self.stdout.write(self.style.SUCCESS(
                f"Exported {len(df)} records to {output_path}\n"
                f"Sample data:\n{df.head()}"
            ))
        else:
            self.stdout.write(self.style.ERROR("No data found to export!"))