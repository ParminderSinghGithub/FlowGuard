import os
import logging
import joblib
import numpy as np
from django.core.cache import cache
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated, AllowAny
from .models import User, TrafficData, CongestionPrediction, PotholeReport, Notification, Route
from .serializers import (
    UserSerializer, TrafficDataSerializer, CongestionPredictionSerializer,
    PotholeReportSerializer, NotificationSerializer, RouteSerializer,
    SensorIngestSerializer, NearbyPotholeQuerySerializer, RouteWarningQuerySerializer,
    PotholeClusterSerializer,
    RouteOptimizationRequestSerializer, RouteRiskAnalysisSerializer,
)
from .traffic_apis.tomtom import get_ludhiana_traffic
from .services.pothole_service import PotholeService
from .services.route_service import RouteIntelligenceService
from .services.background import run_task_with_fallback
from .throttles import SensorIngestUserRateThrottle, SensorIngestAnonRateThrottle, RoutingRateThrottle
from .tasks import verify_pothole_clusters_task

logger = logging.getLogger(__name__)

_MODEL_ASSETS = {
    'ready': False,
    'error': None,
    'interpreter': None,
    'input_details': None,
    'output_details': None,
    'scaler': None,
}


def _load_model_assets():
    """Lazy model/scaler loader to prevent startup-time crashes."""
    if _MODEL_ASSETS['ready']:
        return _MODEL_ASSETS

    if _MODEL_ASSETS['error']:
        return _MODEL_ASSETS

    try:
        import tensorflow as tf

        tflite_model_path = os.path.join(
            os.path.dirname(__file__),
            'tflite_model',
            'models',
            'traffic_lstm_model.tflite',
        )
        scaler_path = os.path.join(os.path.dirname(__file__), 'tflite_model', 'scaler.pkl')

        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()

        _MODEL_ASSETS.update({
            'ready': True,
            'interpreter': interpreter,
            'input_details': interpreter.get_input_details(),
            'output_details': interpreter.get_output_details(),
            'scaler': joblib.load(scaler_path),
        })
    except Exception as exc:
        _MODEL_ASSETS['error'] = str(exc)
        logger.exception('Failed to initialize ML assets: %s', exc)

    return _MODEL_ASSETS

# Ludhiana-specific constants
LUDHIANA_HOTSPOTS = [
    (30.9000, 75.8573),  # City Center
    (30.9158, 75.8227),  # PAU/Sarabha Nagar
    (30.8412, 75.8573),  # Bus Stand
    (30.8786, 75.8000)   # Dugri Rd
]

@csrf_exempt
def predict_traffic(request):
    """Predict congestion for Ludhiana hotspots."""
    if request.method == 'POST':
        try:
            traffic_data = get_ludhiana_traffic()
            if not traffic_data:
                return JsonResponse({"error": "Failed to fetch Ludhiana traffic data"}, status=500)

            speed_ratios = []
            for segment in traffic_data:
                speeds = segment.get('speeds', {})
                free_flow = max(float(speeds.get('free_flow', 0.0)), 1.0)
                current = float(speeds.get('current', 0.0))
                if current < 0:
                    continue
                speed_ratios.append([current / free_flow])

            speed_ratios = speed_ratios[-3:]
            if not speed_ratios:
                return JsonResponse({"error": "No valid traffic segments available"}, status=500)

            while len(speed_ratios) < 3:
                speed_ratios.insert(0, speed_ratios[0])

            assets = _load_model_assets()
            if assets['ready']:
                input_array = np.array(speed_ratios, dtype=np.float32)
                input_array = np.expand_dims(input_array, axis=0)
                assets['interpreter'].set_tensor(assets['input_details'][0]['index'], input_array)
                assets['interpreter'].invoke()
                prediction = assets['interpreter'].get_tensor(assets['output_details'][0]['index'])[0][0]
                denormalized_pred = assets['scaler'].inverse_transform([[prediction]])[0][0]
                model_status = 'ok'
            else:
                denormalized_pred = float(np.mean([v[0] for v in speed_ratios]))
                model_status = f"fallback: {assets['error'] or 'unavailable'}"

            values = [v[0] for v in speed_ratios]
            base = max(float(np.mean(values)), 0.001)
            variability = float(np.std(values)) / base
            prediction_confidence = max(0.1, min(0.99, 1.0 - variability))

            latest_data = TrafficData.objects.order_by('-timestamp').first()
            if latest_data:
                CongestionPrediction.objects.create(
                    location=latest_data,
                    predicted_congestion_level='severe' if denormalized_pred < 0.4 else 'moderate',
                    prediction_time=timezone.now(),
                    accuracy=0.95,
                    prediction_confidence=prediction_confidence,
                )

            return JsonResponse({
                "prediction": float(denormalized_pred),
                "prediction_confidence": prediction_confidence,
                "model_status": model_status,
                "hotspots": [{"lat": lat, "lon": lon} for lat, lon in LUDHIANA_HOTSPOTS]
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "POST method required"}, status=405)

def test_traffic_flow(request):
    """Diagnostic endpoint for TomTom traffic fetch."""
    flow_data = get_ludhiana_traffic()
    return JsonResponse(flow_data, safe=False) if flow_data else \
           JsonResponse({"error": "API failure"}, status=500)

def home(request):
    return HttpResponse("Welcome to FlowGuard App")

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

    @action(detail=True, methods=['get'])
    def routes(self, request, pk=None):
        user = self.get_object()
        routes = user.preferred_routes.all()
        serializer = RouteSerializer(routes, many=True)
        return Response(serializer.data)

class TrafficDataViewSet(viewsets.ModelViewSet):
    queryset = TrafficData.objects.all()
    serializer_class = TrafficDataSerializer

    @action(detail=False, methods=['get'])
    def location_data(self, request):
        latitude = request.query_params.get('latitude')
        longitude = request.query_params.get('longitude')
        if latitude and longitude:
            traffic_data = TrafficData.objects.filter(latitude=latitude, longitude=longitude)
            serializer = self.get_serializer(traffic_data, many=True)
            return Response(serializer.data)
        return Response({"error": "Location parameters missing."}, status=status.HTTP_400_BAD_REQUEST)

class CongestionPredictionViewSet(viewsets.ModelViewSet):
    queryset = CongestionPrediction.objects.all()
    serializer_class = CongestionPredictionSerializer

    @action(detail=False, methods=['get'])
    def location_prediction(self, request):
        location_id = request.query_params.get('location_id')
        if location_id:
            predictions = CongestionPrediction.objects.filter(location_id=location_id)
            serializer = self.get_serializer(predictions, many=True)
            return Response(serializer.data)
        return Response({"error": "Location ID missing."}, status=status.HTTP_400_BAD_REQUEST)

class PotholeReportViewSet(viewsets.ModelViewSet):
    queryset = PotholeReport.objects.all()
    serializer_class = PotholeReportSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['get'])
    def verified_potholes(self, request):
        verified_potholes = PotholeReport.objects.filter(is_verified=True)
        serializer = self.get_serializer(verified_potholes, many=True)
        return Response(serializer.data)

class NotificationViewSet(viewsets.ModelViewSet):
    queryset = Notification.objects.all()
    serializer_class = NotificationSerializer

    @action(detail=True, methods=['post'])
    def mark_as_read(self, request, pk=None):
        notification = self.get_object()
        notification.is_read = True
        notification.save()
        return Response({"status": "Notification marked as read."})

class RouteViewSet(viewsets.ModelViewSet):
    queryset = Route.objects.all()
    serializer_class = RouteSerializer

    @action(detail=True, methods=['get'])
    def traffic_data(self, request, pk=None):
        route = self.get_object()
        traffic_data = route.traffic_data.all()
        serializer = TrafficDataSerializer(traffic_data, many=True)
        return Response(serializer.data)


class SensorDataIngestAPIView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [SensorIngestUserRateThrottle, SensorIngestAnonRateThrottle]
    service = PotholeService()

    def post(self, request):
        limited = self._sensor_ingest_limit(request)
        if limited:
            return Response(
                {'error': {'code': 'rate_limited', 'message': 'Sensor ingestion rate limit exceeded.'}},
                status=status.HTTP_429_TOO_MANY_REQUESTS,
            )

        serializer = SensorIngestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        payload['user_id'] = request.user.id
        result = self.service.ingest_sensor_point(**payload)
        return Response(result, status=status.HTTP_201_CREATED)

    def _sensor_ingest_limit(self, request):
        limit = int(getattr(settings, 'SENSOR_INGEST_RATE_LIMIT', 30))
        window_seconds = int(getattr(settings, 'SENSOR_INGEST_RATE_PERIOD_SECONDS', 60))
        key = f"sensor-ingest:{getattr(request.user, 'id', 'anon')}:{self._client_ip(request)}"
        state = cache.get(key)
        now = timezone.now().timestamp()

        if not state or now - state['window_start'] >= window_seconds:
            cache.set(key, {'window_start': now, 'count': 1}, timeout=window_seconds)
            return False

        if state['count'] >= limit:
            return True

        state['count'] += 1
        cache.set(key, state, timeout=window_seconds)
        return False

    def _client_ip(self, request):
        forwarded = request.META.get('HTTP_X_FORWARDED_FOR')
        if forwarded:
            return forwarded.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR', 'unknown')


class ManualPotholeReportAPIView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [SensorIngestUserRateThrottle, SensorIngestAnonRateThrottle]
    service = PotholeService()

    def post(self, request):
        serializer = PotholeReportSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        report, cluster, verified = self.service.report_manual_pothole(
            latitude=payload['latitude'],
            longitude=payload['longitude'],
            user_id=payload.get('user').id if payload.get('user') else None,
            severity=payload.get('severity', 'moderate'),
            confidence_score=payload.get('confidence_score', 0.5),
        )
        return Response(
            {
                'report': PotholeReportSerializer(report).data,
                'cluster': PotholeClusterSerializer(cluster).data,
                'cluster_verified': verified,
            },
            status=status.HTTP_201_CREATED,
        )


class NearbyPotholesAPIView(APIView):
    permission_classes = [AllowAny]
    service = PotholeService()

    def get(self, request):
        serializer = NearbyPotholeQuerySerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)
        params = serializer.validated_data
        clusters = self.service.nearby_potholes(
            latitude=params['latitude'],
            longitude=params['longitude'],
            radius_meters=params['radius_meters'],
            verified_only=params['verified_only'],
        )
        return Response(PotholeClusterSerializer(clusters, many=True).data)


class VerifyPotholeClustersAPIView(APIView):
    permission_classes = [IsAuthenticated]
    service = PotholeService()

    def post(self, request):
        dispatch = run_task_with_fallback(
            verify_pothole_clusters_task,
            self.service.verify_all_clusters,
        )
        return Response({'dispatch': dispatch})


class RouteWarningsAPIView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [RoutingRateThrottle]
    service = PotholeService()

    def get(self, request):
        serializer = RouteWarningQuerySerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)
        params = serializer.validated_data
        warnings = self.service.route_warnings(
            latitude=params['latitude'],
            longitude=params['longitude'],
            radius_meters=params['radius_meters'],
        )
        return Response({'warnings': warnings, 'warning_count': len(warnings)})


class PredictionModelHealthAPIView(APIView):
    permission_classes = [AllowAny]
    def get(self, request):
        assets = _load_model_assets()
        if assets['ready']:
            return Response({'model_ready': True, 'status': 'ok'})
        return Response(
            {
                'model_ready': False,
                'status': 'degraded',
                'error': assets['error'] or 'unknown',
            },
            status=status.HTTP_200_OK,
        )


class RouteOptimizeAPIView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [RoutingRateThrottle]
    service = RouteIntelligenceService()

    def post(self, request):
        serializer = RouteOptimizationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        params = serializer.validated_data
        result = self.service.optimize(
            start_coords=(params['start_latitude'], params['start_longitude']),
            end_coords=(params['end_latitude'], params['end_longitude']),
            departure_time=params.get('departure_time'),
            eta_tolerance_ratio=params['eta_tolerance_ratio'],
            alternatives_count=params['alternatives_count'],
        )
        if result.get('error'):
            return Response(result, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        return Response(result)


class RouteAlternativesAPIView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [RoutingRateThrottle]
    service = RouteIntelligenceService()

    def post(self, request):
        serializer = RouteOptimizationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        params = serializer.validated_data
        result = self.service.optimize(
            start_coords=(params['start_latitude'], params['start_longitude']),
            end_coords=(params['end_latitude'], params['end_longitude']),
            departure_time=params.get('departure_time'),
            eta_tolerance_ratio=params['eta_tolerance_ratio'],
            alternatives_count=max(params['alternatives_count'], 2),
        )
        if result.get('error'):
            return Response(result, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        return Response({'alternatives': result['alternatives']})


class RouteRiskAnalysisAPIView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [RoutingRateThrottle]
    service = RouteIntelligenceService()

    def post(self, request):
        serializer = RouteRiskAnalysisSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        params = serializer.validated_data
        result = self.service.risk_analysis(
            start_coords=(params['start_latitude'], params['start_longitude']),
            end_coords=(params['end_latitude'], params['end_longitude']),
        )
        if result.get('error'):
            return Response(result, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        return Response(result)
