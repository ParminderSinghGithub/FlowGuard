from django.urls import path, include
from . import views
from rest_framework.routers import DefaultRouter
from .views import UserViewSet, TrafficDataViewSet, CongestionPredictionViewSet, PotholeReportViewSet, NotificationViewSet, RouteViewSet

router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'traffic', TrafficDataViewSet)
router.register(r'predictions', CongestionPredictionViewSet)
router.register(r'potholes', PotholeReportViewSet)
router.register(r'notifications', NotificationViewSet)
router.register(r'routes', RouteViewSet)

urlpatterns = [
    path('', views.home, name='home'),
    path('', include(router.urls)),
    path('predict/', views.predict_traffic, name='predict_traffic'),
]
