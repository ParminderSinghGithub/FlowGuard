from django.urls import path, include
from . import views
from rest_framework.routers import DefaultRouter
from rest_framework.authtoken.views import obtain_auth_token
from .views import (
    UserViewSet,
    UserRegistrationAPIView,
    CustomAuthTokenView,
    TrafficDataViewSet,
    CongestionPredictionViewSet,
    PotholeReportViewSet,
    NotificationViewSet,
    RouteViewSet,
    SensorDataIngestAPIView,
    ManualPotholeReportAPIView,
    NearbyPotholesAPIView,
    VerifyPotholeClustersAPIView,
    RouteWarningsAPIView,
    PredictionModelHealthAPIView,
    RouteOptimizeAPIView,
    RouteAlternativesAPIView,
    RouteRiskAnalysisAPIView,
    PredictTrafficAPIView,
)

router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'traffic', TrafficDataViewSet)
router.register(r'predictions', CongestionPredictionViewSet)
router.register(r'potholes', PotholeReportViewSet)
router.register(r'notifications', NotificationViewSet)
router.register(r'routes', RouteViewSet)

urlpatterns = [
    path('', views.home, name='home'),
    path('auth/register/', UserRegistrationAPIView.as_view(), name='api_user_register'),
    path('auth/token/', CustomAuthTokenView.as_view(), name='api_token_auth'),
    path('predict/', PredictTrafficAPIView.as_view(), name='predict_traffic'),
    path('predict/health/', PredictionModelHealthAPIView.as_view(), name='predict_model_health'),
    path('potholes/sensor/', SensorDataIngestAPIView.as_view(), name='pothole_sensor_ingest'),
    path('potholes/report/', ManualPotholeReportAPIView.as_view(), name='pothole_report_manual'),
    path('potholes/nearby/', NearbyPotholesAPIView.as_view(), name='pothole_nearby'),
    path('potholes/verify-clusters/', VerifyPotholeClustersAPIView.as_view(), name='pothole_verify_clusters'),
    path('routes/warnings/', RouteWarningsAPIView.as_view(), name='route_warnings'),
    path('routes/optimize/', RouteOptimizeAPIView.as_view(), name='route_optimize'),
    path('routes/alternatives/', RouteAlternativesAPIView.as_view(), name='route_alternatives'),
    path('routes/risk-analysis/', RouteRiskAnalysisAPIView.as_view(), name='route_risk_analysis'),
    path('', include(router.urls)),
]
