import json
import logging
import time

logger = logging.getLogger('app.request')
suspicious_logger = logging.getLogger('app.suspicious')


class RequestAuditMiddleware:
    """Lightweight request and suspicious-activity logger."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        started = time.monotonic()
        raw_body = b''
        if hasattr(request, 'body'):
            try:
                raw_body = request.body or b''
            except Exception:
                raw_body = b''

        response = self.get_response(request)
        elapsed_ms = (time.monotonic() - started) * 1000.0

        user_id = getattr(request.user, 'id', None) if hasattr(request, 'user') else None
        logger.info(
            'method=%s path=%s status=%s elapsed_ms=%.2f user_id=%s',
            request.method,
            request.path,
            response.status_code,
            elapsed_ms,
            user_id,
        )

        self._log_suspicious(request, response.status_code, raw_body)
        return response

    def _log_suspicious(self, request, status_code, raw_body):
        body_length = len(raw_body or b'')
        if status_code == 429:
            suspicious_logger.warning('rate_limit_triggered path=%s ip=%s', request.path, self._client_ip(request))

        if request.path.startswith('/api/potholes/sensor/') and body_length > 4096:
            suspicious_logger.warning(
                'large_sensor_payload path=%s ip=%s bytes=%s',
                request.path,
                self._client_ip(request),
                body_length,
            )

        if request.path.startswith('/api/potholes/sensor/') and request.method == 'POST':
            try:
                payload = json.loads(raw_body.decode('utf-8')) if raw_body else {}
            except Exception:
                suspicious_logger.warning('invalid_json_payload path=%s ip=%s', request.path, self._client_ip(request))
                return

            lat = payload.get('latitude')
            lon = payload.get('longitude')
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                if lat < -90 or lat > 90 or lon < -180 or lon > 180:
                    suspicious_logger.warning(
                        'invalid_coordinate_injection path=%s ip=%s latitude=%s longitude=%s',
                        request.path,
                        self._client_ip(request),
                        lat,
                        lon,
                    )

    def _client_ip(self, request):
        forwarded = request.META.get('HTTP_X_FORWARDED_FOR')
        if forwarded:
            return forwarded.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR')
