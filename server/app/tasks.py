from celery import shared_task
import logging
from django.db import connection

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