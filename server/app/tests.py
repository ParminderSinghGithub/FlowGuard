from django.test import TestCase, override_settings
from rest_framework.test import APIClient
from unittest.mock import patch

from app.models import PotholeCluster, PotholeReport, User, TrafficData
from app.services.pothole_service import PotholeService


@override_settings(
	POTHOLE_Z_THRESHOLD=2.0,
	POTHOLE_CLUSTER_RADIUS_METERS=60.0,
	POTHOLE_VERIFY_MIN_REPORTS=3,
	POTHOLE_VERIFY_MIN_DEVICES=2,
)
class PotholeServiceTests(TestCase):
	def setUp(self):
		self.service = PotholeService()

	def test_sensor_spike_creates_candidate_report(self):
		self.service.ingest_sensor_point(
			device_id='dev-1',
			latitude=30.9001,
			longitude=75.8572,
			accelerometer_z=0.5,
		)
		result = self.service.ingest_sensor_point(
			device_id='dev-1',
			latitude=30.9001,
			longitude=75.8572,
			accelerometer_z=5.0,
		)
		self.assertTrue(result['pothole_candidate_created'])
		self.assertEqual(PotholeReport.objects.count(), 1)

	def test_multi_device_cluster_becomes_verified(self):
		samples = [
			('dev-a', 5.2),
			('dev-b', 4.8),
			('dev-c', 4.9),
		]

		for device, z_axis in samples:
			self.service.ingest_sensor_point(
				device_id=device,
				latitude=30.9005,
				longitude=75.8575,
				accelerometer_z=z_axis,
			)

		cluster = PotholeCluster.objects.first()
		self.assertIsNotNone(cluster)
		self.assertTrue(cluster.is_verified)
		self.assertGreaterEqual(cluster.reports_count, 3)


@override_settings(
	POTHOLE_Z_THRESHOLD=2.0,
	POTHOLE_CLUSTER_RADIUS_METERS=60.0,
	POTHOLE_VERIFY_MIN_REPORTS=3,
	POTHOLE_VERIFY_MIN_DEVICES=2,
)
class PotholeApiTests(TestCase):
	def setUp(self):
		self.client = APIClient()
		self.user1 = User.objects.create_user(username='u1', password='x', device_id='d1')
		self.user2 = User.objects.create_user(username='u2', password='x', device_id='d2')
		self.user3 = User.objects.create_user(username='u3', password='x', device_id='d3')

	def test_sensor_endpoint_creates_report(self):
		baseline_payload = {
			'device_id': 'd1',
			'user_id': self.user1.id,
			'latitude': 30.901,
			'longitude': 75.852,
			'accelerometer_z': 0.4,
		}
		spike_payload = {
			'device_id': 'd1',
			'user_id': self.user1.id,
			'latitude': 30.901,
			'longitude': 75.852,
			'accelerometer_z': 5.1,
		}
		self.client.post('/api/potholes/sensor/', baseline_payload, format='json')
		response = self.client.post('/api/potholes/sensor/', spike_payload, format='json')
		self.assertEqual(response.status_code, 201)
		self.assertTrue(response.data['pothole_candidate_created'])

	def test_verify_and_route_warnings(self):
		events = [
			{'device_id': 'd1', 'user_id': self.user1.id, 'latitude': 30.902, 'longitude': 75.854, 'accelerometer_z': 5.3},
			{'device_id': 'd2', 'user_id': self.user2.id, 'latitude': 30.9021, 'longitude': 75.8541, 'accelerometer_z': 5.0},
			{'device_id': 'd3', 'user_id': self.user3.id, 'latitude': 30.9022, 'longitude': 75.8542, 'accelerometer_z': 4.9},
		]

		self.client.post('/api/potholes/sensor/', {
			'device_id': 'd1', 'user_id': self.user1.id, 'latitude': 30.902, 'longitude': 75.854, 'accelerometer_z': 0.3
		}, format='json')
		self.client.post('/api/potholes/sensor/', {
			'device_id': 'd2', 'user_id': self.user2.id, 'latitude': 30.9021, 'longitude': 75.8541, 'accelerometer_z': 0.2
		}, format='json')
		self.client.post('/api/potholes/sensor/', {
			'device_id': 'd3', 'user_id': self.user3.id, 'latitude': 30.9022, 'longitude': 75.8542, 'accelerometer_z': 0.3
		}, format='json')

		for payload in events:
			self.client.post('/api/potholes/sensor/', payload, format='json')

		verify_response = self.client.post('/api/potholes/verify-clusters/', {}, format='json')
		self.assertEqual(verify_response.status_code, 200)

		warnings_response = self.client.get('/api/routes/warnings/?latitude=30.902&longitude=75.854&radius_meters=500')
		self.assertEqual(warnings_response.status_code, 200)
		self.assertGreaterEqual(warnings_response.data['warning_count'], 1)


class PredictionApiTests(TestCase):
	def setUp(self):
		self.client = APIClient()
		TrafficData.objects.create(
			location='Ludhiana',
			latitude=30.9,
			longitude=75.85,
			congestion_level='moderate',
			current_speed=20.0,
			free_flow_speed=40.0,
		)

	def test_model_health_endpoint(self):
		response = self.client.get('/api/predict/health/')
		self.assertEqual(response.status_code, 200)
		self.assertIn('model_ready', response.data)

	@patch('app.views.get_ludhiana_traffic')
	@patch('app.views._load_model_assets')
	def test_predict_traffic_fallback(self, mocked_assets, mocked_traffic):
		mocked_assets.return_value = {
			'ready': False,
			'error': 'model not available',
		}
		mocked_traffic.return_value = [
			{'speeds': {'current': 20.0, 'free_flow': 40.0}},
			{'speeds': {'current': 24.0, 'free_flow': 40.0}},
			{'speeds': {'current': 28.0, 'free_flow': 40.0}},
		]

		response = self.client.post('/api/predict/', {}, format='json')
		self.assertEqual(response.status_code, 200)
		payload = response.json()
		self.assertIn('prediction', payload)
		self.assertIn('prediction_confidence', payload)
		self.assertIn('fallback', payload['model_status'])
