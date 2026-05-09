from django.urls import path, include
from . import views
from rest_framework.routers import DefaultRouter
from .views import (
    UserViewSet,
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
    path('predict/', views.predict_traffic, name='predict_traffic'),
    path('predict/health/', PredictionModelHealthAPIView.as_view(), name='predict_model_health'),
    path('test-traffic-flow/', views.test_traffic_flow, name='test_traffic_flow'),
    path('potholes/sensor/', SensorDataIngestAPIView.as_view(), name='pothole_sensor_ingest'),
    path('potholes/report/', ManualPotholeReportAPIView.as_view(), name='pothole_report_manual'),
    path('potholes/nearby/', NearbyPotholesAPIView.as_view(), name='pothole_nearby'),
    path('potholes/verify-clusters/', VerifyPotholeClustersAPIView.as_view(), name='pothole_verify_clusters'),
    path('routes/warnings/', RouteWarningsAPIView.as_view(), name='route_warnings'),
    path('', include(router.urls)),
]
