from django.core.cache import cache
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
		for device, z_axis in [('dev-a', 5.2), ('dev-b', 4.8), ('dev-c', 4.9)]:
			self.service.ingest_sensor_point(
				device_id=device,
				latitude=30.9005,
				longitude=75.8575,
				accelerometer_z=z_axis,
			)

		cluster = PotholeCluster.objects.first()
		self.assertIsNotNone(cluster)
		self.assertTrue(cluster.is_verified)


@override_settings(
	POTHOLE_Z_THRESHOLD=2.0,
	POTHOLE_CLUSTER_RADIUS_METERS=60.0,
	POTHOLE_VERIFY_MIN_REPORTS=3,
	POTHOLE_VERIFY_MIN_DEVICES=2,
)
class SecurityAndPotholeApiTests(TestCase):
	def setUp(self):
		self.client = APIClient()
		self.user1 = User.objects.create_user(username='u1', password='x', device_id='d1')
		self.user2 = User.objects.create_user(username='u2', password='x', device_id='d2')
		self.user3 = User.objects.create_user(username='u3', password='x', device_id='d3')

	def _auth(self, user):
		self.client.force_authenticate(user=user)

	def test_sensor_endpoint_requires_auth(self):
		response = self.client.post(
			'/api/potholes/sensor/',
			{'device_id': 'd1', 'latitude': 30.9, 'longitude': 75.85, 'accelerometer_z': 1.0},
			format='json',
		)
		self.assertEqual(response.status_code, 401)

	def test_token_auth_endpoint_issues_token(self):
		response = self.client.post('/api/auth/token/', {'username': 'u1', 'password': 'x'}, format='json')
		self.assertEqual(response.status_code, 200)
		self.assertIn('token', response.data)

	def test_signup_requires_matching_password_confirmation(self):
		response = self.client.post(
			'/api/auth/register/',
			{
				'username': 'new-user',
				'password': 'StrongPass123!',
				'password_confirmation': 'different',
			},
			format='json',
		)
		self.assertEqual(response.status_code, 400)
		self.assertIn('password_confirmation', response.data['errors'])

	def test_signup_rejects_duplicate_username(self):
		response = self.client.post(
			'/api/auth/register/',
			{
				'username': 'U1',
				'password': 'StrongPass123!',
				'password_confirmation': 'StrongPass123!',
			},
			format='json',
		)
		self.assertEqual(response.status_code, 400)
		self.assertIn('username', response.data['errors'])

	def test_signup_issues_token(self):
		response = self.client.post(
			'/api/auth/register/',
			{
				'username': 'fresh-user',
				'password': 'StrongPass123!',
				'password_confirmation': 'StrongPass123!',
			},
			format='json',
		)
		self.assertEqual(response.status_code, 201)
		self.assertIn('token', response.data)

	def test_sensor_endpoint_rejects_invalid_coordinates(self):
		self._auth(self.user1)
		response = self.client.post(
			'/api/potholes/sensor/',
			{'device_id': 'd1', 'latitude': 301.0, 'longitude': 75.85, 'accelerometer_z': 3.0},
			format='json',
		)
		self.assertEqual(response.status_code, 400)

	@override_settings(
		SENSOR_INGEST_RATE_LIMIT=2,
		SENSOR_INGEST_RATE_PERIOD_SECONDS=60,
		REST_FRAMEWORK={
			'DEFAULT_AUTHENTICATION_CLASSES': (
				'rest_framework.authentication.TokenAuthentication',
				'rest_framework.authentication.SessionAuthentication',
			),
			'DEFAULT_PERMISSION_CLASSES': ('rest_framework.permissions.AllowAny',),
			'DEFAULT_THROTTLE_CLASSES': (
				'rest_framework.throttling.UserRateThrottle',
				'rest_framework.throttling.AnonRateThrottle',
			),
			'DEFAULT_THROTTLE_RATES': {
				'user': '120/min',
				'anon': '60/min',
				'sensor_ingest_user': '2/min',
				'sensor_ingest_anon': '1/min',
				'routing_api': '50/min',
			},
			'EXCEPTION_HANDLER': 'app.exceptions.standardized_exception_handler',
		}
	)
	def test_drf_sensor_rate_limiting(self):
		cache.clear()
		self._auth(self.user1)
		payload = {'device_id': 'd1', 'latitude': 30.9, 'longitude': 75.85, 'accelerometer_z': 0.2}
		self.client.post('/api/potholes/sensor/', payload, format='json')
		self.client.post('/api/potholes/sensor/', payload, format='json')
		response = self.client.post('/api/potholes/sensor/', payload, format='json')
		self.assertEqual(response.status_code, 429)

	@override_settings(SENSOR_INGEST_RATE_LIMIT=2, SENSOR_INGEST_RATE_PERIOD_SECONDS=60)
	def test_sensor_rate_limiting(self):
		cache.clear()
		self._auth(self.user1)
		payload = {'device_id': 'd1', 'latitude': 30.9, 'longitude': 75.85, 'accelerometer_z': 0.2}
		self.client.post('/api/potholes/sensor/', payload, format='json')
		self.client.post('/api/potholes/sensor/', payload, format='json')
		response = self.client.post('/api/potholes/sensor/', payload, format='json')
		self.assertEqual(response.status_code, 429)

	def test_verify_and_route_warnings(self):
		self._auth(self.user1)
		# Establish baselines
		self.client.post('/api/potholes/sensor/', {'device_id': 'd1', 'latitude': 30.902, 'longitude': 75.854, 'accelerometer_z': 0.3}, format='json')

		self.client.force_authenticate(user=self.user2)
		self.client.post('/api/potholes/sensor/', {'device_id': 'd2', 'latitude': 30.9021, 'longitude': 75.8541, 'accelerometer_z': 0.2}, format='json')

		self.client.force_authenticate(user=self.user3)
		self.client.post('/api/potholes/sensor/', {'device_id': 'd3', 'latitude': 30.9022, 'longitude': 75.8542, 'accelerometer_z': 0.3}, format='json')

		for user, payload in [
			(self.user1, {'device_id': 'd1', 'latitude': 30.902, 'longitude': 75.854, 'accelerometer_z': 5.3}),
			(self.user2, {'device_id': 'd2', 'latitude': 30.9021, 'longitude': 75.8541, 'accelerometer_z': 5.0}),
			(self.user3, {'device_id': 'd3', 'latitude': 30.9022, 'longitude': 75.8542, 'accelerometer_z': 4.9}),
		]:
			self.client.force_authenticate(user=user)
			self.client.post('/api/potholes/sensor/', payload, format='json')

		self.client.force_authenticate(user=self.user1)
		verify_response = self.client.post('/api/potholes/verify-clusters/', {}, format='json')
		self.assertEqual(verify_response.status_code, 200)

		warnings_response = self.client.get('/api/routes/warnings/?latitude=30.902&longitude=75.854&radius_meters=500')
		self.assertEqual(warnings_response.status_code, 200)


class RouteIntelligenceApiTests(TestCase):
	def setUp(self):
		self.client = APIClient()
		self.user = User.objects.create_user(username='route-user', password='x', device_id='route-device')
		self.client.force_authenticate(user=self.user)

		cluster = PotholeCluster.objects.create(
			centroid_latitude=30.915,
			centroid_longitude=75.855,
			reports_count=5,
			confidence_aggregate=0.9,
			is_verified=True,
		)
		PotholeReport.objects.create(
			user=self.user,
			source_device_id='route-device',
			latitude=30.915,
			longitude=75.855,
			severity='severe',
			source_type='manual',
			confidence_score=0.9,
			is_verified=True,
			cluster=cluster,
		)

	def test_optimize_route_with_risk_outputs(self):
		payload = {
			'start_latitude': 30.911972,
			'start_longitude': 75.853222,
			'end_latitude': 30.900000,
			'end_longitude': 75.840000,
			'alternatives_count': 3,
		}
		response = self.client.post('/api/routes/optimize/', payload, format='json')
		self.assertEqual(response.status_code, 200)
		self.assertIn('selected_route', response.data)
		self.assertIn('route_risk_score', response.data['selected_route'])
		self.assertIn('pothole_warning_count', response.data['selected_route'])

	def test_alternatives_and_risk_analysis(self):
		payload = {
			'start_latitude': 30.911972,
			'start_longitude': 75.853222,
			'end_latitude': 30.900000,
			'end_longitude': 75.840000,
		}
		alternatives = self.client.post('/api/routes/alternatives/', payload, format='json')
		self.assertEqual(alternatives.status_code, 200)
		self.assertIn('alternatives', alternatives.data)

		risk = self.client.post('/api/routes/risk-analysis/', payload, format='json')
		self.assertEqual(risk.status_code, 200)
		self.assertIn('route_risk_score', risk.data)


class BackgroundFallbackTests(TestCase):
	def setUp(self):
		self.client = APIClient()
		self.user = User.objects.create_user(username='bg-user', password='x', device_id='bg-device')
		self.client.force_authenticate(user=self.user)

	@patch('app.views.verify_pothole_clusters_task.delay', side_effect=RuntimeError('broker down'))
	@patch('app.views.PotholeService.verify_all_clusters', return_value=[11, 22])
	def test_verify_clusters_sync_fallback_when_broker_fails(self, mocked_sync, mocked_delay):
		response = self.client.post('/api/potholes/verify-clusters/', {}, format='json')
		self.assertEqual(response.status_code, 200)
		self.assertEqual(response.data['dispatch']['mode'], 'sync_fallback')
		mocked_delay.assert_called_once()
		mocked_sync.assert_called_once()


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
		mocked_assets.return_value = {'ready': False, 'error': 'model not available'}
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
