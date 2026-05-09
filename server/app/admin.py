from django.contrib import admin
from .models import (
	User,
	TrafficData,
	CongestionPrediction,
	Route,
	PotholeReport,
	Notification,
	PotholeCluster,
	SensorDataPoint,
	UserLocation,
)

# Register your models here.
admin.site.register(User)
admin.site.register(TrafficData)
admin.site.register(CongestionPrediction)
admin.site.register(Route)
admin.site.register(PotholeReport)
admin.site.register(Notification)
admin.site.register(PotholeCluster)
admin.site.register(SensorDataPoint)
admin.site.register(UserLocation)
