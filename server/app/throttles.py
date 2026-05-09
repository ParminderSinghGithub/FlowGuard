from rest_framework.throttling import UserRateThrottle, AnonRateThrottle


class SensorIngestUserRateThrottle(UserRateThrottle):
    scope = 'sensor_ingest_user'


class SensorIngestAnonRateThrottle(AnonRateThrottle):
    scope = 'sensor_ingest_anon'


class RoutingRateThrottle(UserRateThrottle):
    scope = 'routing_api'
