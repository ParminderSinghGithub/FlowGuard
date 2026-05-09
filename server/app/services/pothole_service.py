from datetime import timedelta

from django.conf import settings
from django.db import transaction
from django.db.models import Avg
from django.utils import timezone

from app.models import (
    Notification,
    PotholeCluster,
    PotholeReport,
    SensorDataPoint,
    User,
    UserLocation,
)
from app.services.geo import haversine_distance_meters


class PotholeService:
    def __init__(self):
        self.z_threshold = float(getattr(settings, 'POTHOLE_Z_THRESHOLD', 3.2))
        self.debounce_seconds = int(getattr(settings, 'POTHOLE_DEBOUNCE_SECONDS', 10))
        self.device_cooldown_seconds = int(getattr(settings, 'POTHOLE_DEVICE_COOLDOWN_SECONDS', 60))
        self.cluster_radius_meters = float(getattr(settings, 'POTHOLE_CLUSTER_RADIUS_METERS', 35.0))
        self.verify_min_reports = int(getattr(settings, 'POTHOLE_VERIFY_MIN_REPORTS', 3))
        self.verify_min_devices = int(getattr(settings, 'POTHOLE_VERIFY_MIN_DEVICES', 2))
        self.warning_radius_meters = float(getattr(settings, 'POTHOLE_WARNING_RADIUS_METERS', 500.0))

    @transaction.atomic
    def ingest_sensor_point(self, *, device_id, latitude, longitude, accelerometer_z, user_id=None):
        user = None
        if user_id:
            user = User.objects.filter(id=user_id).first()

        baseline = self._device_baseline(device_id)

        point = SensorDataPoint.objects.create(
            user=user,
            device_id=device_id,
            latitude=latitude,
            longitude=longitude,
            accelerometer_z=accelerometer_z,
        )

        if user:
            UserLocation.objects.update_or_create(
                user=user,
                defaults={'latitude': latitude, 'longitude': longitude},
            )

        delta = abs(accelerometer_z - baseline)
        is_spike = delta >= self.z_threshold

        if not is_spike:
            return {
                'point_id': point.id,
                'pothole_candidate_created': False,
                'reason': 'below_threshold',
                'baseline_z': baseline,
                'delta': delta,
            }

        if self._recent_duplicate_for_device(device_id, latitude, longitude):
            return {
                'point_id': point.id,
                'pothole_candidate_created': False,
                'reason': 'cooldown_duplicate',
                'baseline_z': baseline,
                'delta': delta,
            }

        confidence = min(1.0, delta / max(self.z_threshold, 0.1))
        severity = self._severity_from_confidence(confidence)

        report = PotholeReport.objects.create(
            user=user,
            source_device_id=device_id,
            latitude=latitude,
            longitude=longitude,
            severity=severity,
            source_type='sensor',
            accelerometer_z=accelerometer_z,
            confidence_score=confidence,
        )

        cluster = self._attach_to_cluster(report)
        verified = self._verify_cluster(cluster)

        return {
            'point_id': point.id,
            'pothole_candidate_created': True,
            'report_id': report.id,
            'cluster_id': cluster.id,
            'cluster_verified': verified,
            'confidence': confidence,
            'baseline_z': baseline,
            'delta': delta,
        }

    @transaction.atomic
    def report_manual_pothole(self, *, latitude, longitude, user_id=None, severity='moderate', confidence_score=0.5):
        user = User.objects.filter(id=user_id).first() if user_id else None
        report = PotholeReport.objects.create(
            user=user,
            source_device_id=user.device_id if user else '',
            latitude=latitude,
            longitude=longitude,
            severity=severity,
            source_type='manual',
            confidence_score=max(0.0, min(1.0, confidence_score)),
        )
        cluster = self._attach_to_cluster(report)
        verified = self._verify_cluster(cluster)
        return report, cluster, verified

    def nearby_potholes(self, *, latitude, longitude, radius_meters=250.0, verified_only=True):
        qs = PotholeCluster.objects.all()
        if verified_only:
            qs = qs.filter(is_verified=True)

        items = []
        for cluster in qs:
            distance_m = haversine_distance_meters(
                latitude,
                longitude,
                cluster.centroid_latitude,
                cluster.centroid_longitude,
            )
            if distance_m <= radius_meters:
                items.append((distance_m, cluster))

        items.sort(key=lambda x: x[0])
        return [cluster for _, cluster in items]

    @transaction.atomic
    def verify_all_clusters(self):
        verified_ids = []
        for cluster in PotholeCluster.objects.all():
            if self._verify_cluster(cluster):
                verified_ids.append(cluster.id)
        return verified_ids

    def route_warnings(self, *, latitude, longitude, radius_meters=500.0):
        clusters = self.nearby_potholes(
            latitude=latitude,
            longitude=longitude,
            radius_meters=radius_meters,
            verified_only=True,
        )
        warnings = []
        for cluster in clusters:
            warnings.append({
                'cluster_id': cluster.id,
                'latitude': cluster.centroid_latitude,
                'longitude': cluster.centroid_longitude,
                'reports_count': cluster.reports_count,
                'confidence_aggregate': cluster.confidence_aggregate,
                'warning': 'Verified pothole nearby. Route will be deprioritized if alternatives exist.',
            })
        return warnings

    def _device_baseline(self, device_id):
        since = timezone.now() - timedelta(minutes=5)
        recent_points = SensorDataPoint.objects.filter(
            device_id=device_id,
            recorded_at__gte=since,
        ).order_by('-recorded_at')[:20]

        values = [p.accelerometer_z for p in recent_points]
        if not values:
            return 0.0

        values.sort()
        mid = len(values) // 2
        if len(values) % 2 == 0:
            return (values[mid - 1] + values[mid]) / 2.0
        return values[mid]

    def _recent_duplicate_for_device(self, device_id, latitude, longitude):
        since = timezone.now() - timedelta(seconds=self.device_cooldown_seconds)
        recent_reports = PotholeReport.objects.filter(
            source_device_id=device_id,
            timestamp__gte=since,
        )
        for report in recent_reports:
            distance = haversine_distance_meters(latitude, longitude, report.latitude, report.longitude)
            if distance <= self.cluster_radius_meters:
                return True
        return False

    def _severity_from_confidence(self, confidence):
        if confidence >= 0.9:
            return 'severe'
        if confidence >= 0.6:
            return 'moderate'
        return 'minor'

    def _attach_to_cluster(self, report):
        candidate_cluster = None
        min_distance = None

        for cluster in PotholeCluster.objects.all():
            distance = haversine_distance_meters(
                report.latitude,
                report.longitude,
                cluster.centroid_latitude,
                cluster.centroid_longitude,
            )
            if distance <= self.cluster_radius_meters and (min_distance is None or distance < min_distance):
                min_distance = distance
                candidate_cluster = cluster

        if candidate_cluster is None:
            candidate_cluster = PotholeCluster.objects.create(
                centroid_latitude=report.latitude,
                centroid_longitude=report.longitude,
                radius_meters=self.cluster_radius_meters,
                reports_count=0,
                confidence_aggregate=0.0,
            )

        report.cluster = candidate_cluster
        report.save(update_fields=['cluster'])
        self._refresh_cluster(candidate_cluster)
        return candidate_cluster

    def _refresh_cluster(self, cluster):
        reports = cluster.reports.all()
        cluster.reports_count = reports.count()
        cluster.confidence_aggregate = reports.aggregate(avg=Avg('confidence_score')).get('avg') or 0.0

        if cluster.reports_count > 0:
            cluster.centroid_latitude = sum(r.latitude for r in reports) / cluster.reports_count
            cluster.centroid_longitude = sum(r.longitude for r in reports) / cluster.reports_count

        cluster.save(
            update_fields=[
                'reports_count',
                'confidence_aggregate',
                'centroid_latitude',
                'centroid_longitude',
                'updated_at',
            ]
        )

    def _verify_cluster(self, cluster):
        self._refresh_cluster(cluster)
        unique_devices = set(
            cluster.reports.exclude(source_device_id='').values_list('source_device_id', flat=True)
        )
        unique_users = set(
            cluster.reports.exclude(user_id=None).values_list('user_id', flat=True)
        )
        evidence_count = max(len(unique_devices), len(unique_users))

        should_verify = (
            cluster.reports_count >= self.verify_min_reports
            and evidence_count >= self.verify_min_devices
        )

        if should_verify and not cluster.is_verified:
            cluster.is_verified = True
            cluster.last_verified_at = timezone.now()
            cluster.save(update_fields=['is_verified', 'last_verified_at', 'updated_at'])
            cluster.reports.update(is_verified=True)
            self._notify_nearby_users(cluster)
            return True

        if should_verify:
            cluster.reports.update(is_verified=True)

        return cluster.is_verified

    def _notify_nearby_users(self, cluster):
        nearby_locations = UserLocation.objects.select_related('user').all()
        for location in nearby_locations:
            distance = haversine_distance_meters(
                location.latitude,
                location.longitude,
                cluster.centroid_latitude,
                cluster.centroid_longitude,
            )
            if distance <= self.warning_radius_meters and location.user.is_active_user:
                Notification.objects.create(
                    user=location.user,
                    message=(
                        'Verified pothole detected nearby. '
                        'Your route suggestions may avoid this road segment.'
                    ),
                )
