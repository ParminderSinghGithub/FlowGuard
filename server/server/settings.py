"""Django settings for FlowGuard server."""

# STARTUP DIAGNOSTICS - Identify active settings file
print("ACTIVE SETTINGS FILE:", __file__)
print("DEBUG DJANGO_SETTINGS_MODULE =", os.environ.get("DJANGO_SETTINGS_MODULE"))

from pathlib import Path
from kombu import Queue
from celery.schedules import crontab
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

env_path = BASE_DIR.parent / '.env'
load_dotenv(env_path)

# ===== SECURITY: Validate SECRET_KEY is not using dev default =====
SECRET_KEY = os.getenv('SECRET_KEY')
if not SECRET_KEY or SECRET_KEY == 'dev-only-change-me':
    from django.core.exceptions import ImproperlyConfigured
    raise ImproperlyConfigured(
        'SECRET_KEY environment variable must be set and not use default value. '
        'Generate a secure key for production: python -c "import secrets; print(secrets.token_urlsafe(50))"'
    )

# ===== Runtime mode =====
# DEBUG controls error verbosity. FLOWGUARD_ENV controls deployment security defaults.
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
FLOWGUARD_ENV = os.getenv('FLOWGUARD_ENV', 'development').strip().lower()
IS_PRODUCTION = FLOWGUARD_ENV == 'production'

# HARDCODED ALLOWED_HOSTS - No environment variable parsing
ALLOWED_HOSTS = [
    "flowguard-bznl.onrender.com",
    "flow-guard-kappa.vercel.app",
    "localhost",
    "127.0.0.1",
]

print("DEBUG ALLOWED_HOSTS =", ALLOWED_HOSTS)

SECURE_SSL_REDIRECT = os.getenv('SECURE_SSL_REDIRECT', 'true' if IS_PRODUCTION else 'false').lower() == 'true'
SESSION_COOKIE_SECURE = os.getenv('SESSION_COOKIE_SECURE', 'true' if IS_PRODUCTION else 'false').lower() == 'true'
CSRF_COOKIE_SECURE = os.getenv('CSRF_COOKIE_SECURE', 'true' if IS_PRODUCTION else 'false').lower() == 'true'
SECURE_HSTS_SECONDS = int(os.getenv('SECURE_HSTS_SECONDS', '31536000' if IS_PRODUCTION else '0'))
SECURE_HSTS_INCLUDE_SUBDOMAINS = os.getenv('SECURE_HSTS_INCLUDE_SUBDOMAINS', 'true' if IS_PRODUCTION else 'false').lower() == 'true'
SECURE_HSTS_PRELOAD = os.getenv('SECURE_HSTS_PRELOAD', 'false').lower() == 'true'
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https') if os.getenv('SECURE_PROXY_SSL_HEADER', 'false').lower() == 'true' else None


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders',
    'app.apps.AppConfig',
    'rest_framework',
    'rest_framework.authtoken',
    'django_celery_beat',
    'django_celery_results',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'app.middleware.RequestAuditMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'server.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'server.wsgi.application'


# SQLite-only by project requirement
sqlite_db_path = os.getenv('SQLITE_DB_PATH')
sqlite_db_name = Path(sqlite_db_path) if sqlite_db_path else BASE_DIR / 'db.sqlite3'
if not sqlite_db_name.is_absolute():
    sqlite_db_name = BASE_DIR.parent / sqlite_db_name

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': sqlite_db_name,
    }
}

# Password validation
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

AUTH_USER_MODEL = 'app.User'

TOMTOM_API_KEY = os.getenv('TOMTOM_API_KEY')

# Celery defaults (can be overridden via environment)
CELERY_BROKER_URL                 = os.getenv('CELERY_BROKER_URL', 'amqp://guest:guest@localhost:5672//')
CELERY_RESULT_BACKEND             = os.getenv('CELERY_RESULT_BACKEND', 'rpc://')
CELERY_ACCEPT_CONTENT             = ['json']
CELERY_TASK_SERIALIZER            = 'json'
CELERY_RESULT_EXTENDED            = True
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True
CELERY_WORKER_CANCEL_LONG_RUNNING_TASKS_ON_CONNECTION_LOSS = True
CELERY_BEAT_SCHEDULER             = 'django_celery_beat.schedulers:DatabaseScheduler'
CELERY_TIMEZONE                   = 'Asia/Kolkata'
CELERY_TASK_DEFAULT_QUEUE         = 'default'
CELERY_TASK_QUEUES                = ( Queue('default', routing_key='default'), )
CELERY_TASK_ACKS_LATE             = True
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_BROKER_POOL_LIMIT          = None
CELERY_BROKER_CONNECTION_MAX_RETRIES = 5
CELERY_BROKER_CONNECTION_RETRY    = True
CELERY_BROKER_CONNECTION_TIMEOUT  = 30
CELERY_WORKER_POOL                = 'solo'

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'flowguard-default',
    }
}

CELERY_BEAT_SCHEDULE = {
    'fetch-ludhiana-traffic': {
        'task': 'app.tasks.fetch_ludhiana_traffic',
        'schedule': crontab(minute='*/2'),
    },
    'verify-pothole-clusters': {
        'task': 'app.tasks.verify_pothole_clusters_task',
        'schedule': crontab(minute='*/5'),
    },
    'cleanup-sensor-points': {
        'task': 'app.tasks.cleanup_stale_sensor_points_task',
        'schedule': crontab(hour='*/6', minute='0'),
    },
    'cleanup-notifications': {
        'task': 'app.tasks.cleanup_stale_notifications_task',
        'schedule': crontab(hour='3', minute='15'),
    },
    'cleanup-inactive-clusters': {
        'task': 'app.tasks.cleanup_inactive_clusters_task',
        'schedule': crontab(hour='4', minute='0'),
    },
    'refresh-route-cache': {
        'task': 'app.tasks.refresh_route_cache_task',
        'schedule': crontab(minute='*/15'),
    },
}

POTHOLE_Z_THRESHOLD = float(os.getenv('POTHOLE_Z_THRESHOLD', '3.2'))
POTHOLE_DEBOUNCE_SECONDS = int(os.getenv('POTHOLE_DEBOUNCE_SECONDS', '10'))
POTHOLE_DEVICE_COOLDOWN_SECONDS = int(os.getenv('POTHOLE_DEVICE_COOLDOWN_SECONDS', '60'))
POTHOLE_CLUSTER_RADIUS_METERS = float(os.getenv('POTHOLE_CLUSTER_RADIUS_METERS', '35.0'))
POTHOLE_VERIFY_MIN_REPORTS = int(os.getenv('POTHOLE_VERIFY_MIN_REPORTS', '3'))
POTHOLE_VERIFY_MIN_DEVICES = int(os.getenv('POTHOLE_VERIFY_MIN_DEVICES', '2'))
POTHOLE_WARNING_RADIUS_METERS = float(os.getenv('POTHOLE_WARNING_RADIUS_METERS', '500.0'))
SENSOR_INGEST_RATE_LIMIT = int(os.getenv('SENSOR_INGEST_RATE_LIMIT', '30'))
SENSOR_INGEST_RATE_PERIOD_SECONDS = int(os.getenv('SENSOR_INGEST_RATE_PERIOD_SECONDS', '60'))

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework.authentication.TokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    'DEFAULT_THROTTLE_CLASSES': (
        'rest_framework.throttling.UserRateThrottle',
        'rest_framework.throttling.AnonRateThrottle',
    ),
    'DEFAULT_THROTTLE_RATES': {
        'user': os.getenv('THROTTLE_USER_RATE', '120/min'),
        'anon': os.getenv('THROTTLE_ANON_RATE', '60/min'),
        'sensor_ingest_user': os.getenv('THROTTLE_SENSOR_USER_RATE', '30/min'),
        'sensor_ingest_anon': os.getenv('THROTTLE_SENSOR_ANON_RATE', '10/min'),
        'routing_api': os.getenv('THROTTLE_ROUTING_RATE', '50/min'),
    },
    'EXCEPTION_HANDLER': 'app.exceptions.standardized_exception_handler',
}

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'app.request': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'app.suspicious': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG',
    },
}

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'Asia/Kolkata'

USE_I18N = True

USE_TZ = True


STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static'] if (BASE_DIR / 'static').exists() else []
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
WHITENOISE_USE_FINDERS = DEBUG
WHITENOISE_AUTOREFRESH = DEBUG

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# CORS Configuration
CORS_ALLOWED_ORIGINS = [
    "https://flow-guard-kappa.vercel.app",
]

CORS_ALLOW_CREDENTIALS = True

CSRF_TRUSTED_ORIGINS = [
    "https://flow-guard-kappa.vercel.app",
]
