import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

app = Celery('server')
app.config_from_object('django.conf:settings', namespace='CELERY')

# Explicitly specify which packages to search for tasks
app.autodiscover_tasks(['app'])  # Replace 'app' with your actual app name

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')