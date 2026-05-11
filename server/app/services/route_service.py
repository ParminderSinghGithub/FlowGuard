from datetime import datetime
import logging

from app.models import PotholeCluster
from app.services.geo import haversine_distance_meters
from app.services.guidance_service import SmartGuidanceService

logger = logging.getLogger(__name__)


class RouteIntelligenceService:
    """Pothole-aware route scoring with graceful fallback behavior."""

    def __init__(self):
        self.segment_influence_radius_m = 180.0
        self.base_penalty_seconds = 140.0
        self._optimizer = None
        self._optimizer_error = None
        self._guidance_service = SmartGuidanceService()

    def _get_optimizer(self):
        if self._optimizer:
            return self._optimizer

        if self._optimizer_error:
            raise RuntimeError(self._optimizer_error)

        try:
            from app.tflite_model.ml_pipeline.route_optimizer import RouteOptimizer

            self._optimizer = RouteOptimizer()
            return self._optimizer
        except Exception as exc:
            self._optimizer_error = str(exc)
            logger.exception('Route optimizer initialization failed: %s', exc)
            raise

    def optimize(self, *, start_coords, end_coords, departure_time=None, eta_tolerance_ratio=1.15, alternatives_count=3):
        if departure_time is None:
            departure_time = datetime.now()

        # Try to get ML optimizer, but have multiple fallback levels
        try:
            optimizer = self._get_optimizer()
            graph = optimizer.route_graph
            optimizer_available = True
        except Exception as exc:
            logger.warning('Route optimizer unavailable, using fallback: %s', exc)
            optimizer_available = False
            graph = None

        # If ML optimizer is available, try intelligent routing
        if optimizer_available:
            try:
                start_segment = optimizer._nearest_segment(start_coords)
                end_segment = optimizer._nearest_segment(end_coords)
                if not start_segment or not end_segment:
                    return self._fallback_route_optimization(start_coords, end_coords, 'Could not map coordinates to route graph segments.')

                candidate_paths = self._enumerate_paths(graph, start_segment, end_segment, max_depth=8)
                if not candidate_paths:
                    return self._fallback_route_optimization(start_coords, end_coords, 'No valid route found in graph.')

                scored = []
                for path in candidate_paths:
                    score = self._score_path(graph, path)
                    scored.append(score)

                scored.sort(key=lambda x: x['composite_score'])
                best_by_score = scored[0]
                best_by_eta = min(scored, key=lambda x: x['eta_seconds'])

                # Prefer smoother route when ETA difference remains reasonable.
                selected = best_by_score
                if best_by_score['eta_seconds'] > best_by_eta['eta_seconds'] * eta_tolerance_ratio:
                    selected = best_by_eta

                alternatives = [item for item in scored if item['path'] != selected['path']][:alternatives_count]

                return {
                    'selected_route': self._format_route_result(selected),
                    'alternatives': [self._format_route_result(item) for item in alternatives],
                    'graph_mode': 'local_graph',
                    'fallback_mode': False,
                }
            except Exception as exc:
                logger.warning('ML routing failed, falling back to static routing: %s', exc)
                # Fall back to static routing if ML routing fails
                return self._fallback_route_optimization(start_coords, end_coords, f'ML routing error: {exc}')
        
        # If no optimizer available, use static fallback routing
        return self._fallback_route_optimization(start_coords, end_coords, 'Route engine unavailable')

    def _fallback_route_optimization(self, start_coords, end_coords, error_message):
        """Fallback route optimization when ML components are unavailable."""
        try:
            # Calculate direct distance-based route
            distance = haversine_distance_meters(
                start_coords[0], start_coords[1],
                end_coords[0], end_coords[1]
            )
            
            # Base ETA on reasonable urban speed (25 km/h = ~7 m/s)
            base_eta_seconds = distance / 7.0
            
            # Add some penalty for unknown conditions
            penalty_seconds = min(300, base_eta_seconds * 0.2)  # Max 5 min penalty
            
            # Create a simple route representation
            route_data = {
                'segments': ['fallback_route'],
                'eta_seconds': round(base_eta_seconds + penalty_seconds, 2),
                'pothole_penalty_seconds': round(penalty_seconds, 2),
                'route_risk_score': min(40.0, distance / 100),  # Simple risk based on distance
                'pothole_warning_count': 0,
                'affected_coordinates': [],
                'composite_score': round(base_eta_seconds + penalty_seconds, 2),
            }
            
            return {
                'selected_route': route_data,
                'alternatives': [],
                'graph_mode': 'fallback_direct',
                'fallback_mode': True,
                'error': error_message,
                'details': f'Using direct distance-based routing ({distance:.0f}m)',
            }
        except Exception as exc:
            logger.error('Fallback routing failed: %s', exc)
            return {
                'error': 'Route optimization completely failed.',
                'fallback_mode': True,
                'details': f'Fallback error: {exc}',
            }

    def risk_analysis(self, *, start_coords, end_coords):
        result = self.optimize(start_coords=start_coords, end_coords=end_coords, alternatives_count=1)
        if result.get('error'):
            return result

        selected = result['selected_route']
        return {
            'route_risk_score': selected['route_risk_score'],
            'pothole_warning_count': selected['pothole_warning_count'],
            'affected_coordinates': selected['affected_coordinates'],
            'penalty_seconds': selected['pothole_penalty_seconds'],
            'eta_seconds': selected['eta_seconds'],
        }

    def _enumerate_paths(self, graph, start, end, max_depth=8):
        paths = []

        def dfs(current, target, visited, path):
            if len(path) > max_depth:
                return
            if current == target:
                paths.append(path[:])
                return

            neighbors = set(graph[current].get('connects_to', []))
            for seg_id, data in graph.items():
                if current in data.get('connects_to', []):
                    neighbors.add(seg_id)

            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, target, visited, path)
                path.pop()
                visited.remove(neighbor)

        dfs(start, end, {start}, [start])
        return paths

    def _score_path(self, graph, path):
        eta_seconds = 0.0
        affected_coordinates = []
        pothole_warning_count = 0
        pothole_penalty_seconds = 0.0

        verified_clusters = list(PotholeCluster.objects.filter(is_verified=True))

        for segment_id in path:
            segment = graph[segment_id]
            speed = max(segment.get('historical_speed', 25.0), 5.0)
            eta_seconds += segment['length'] / (speed * 1000.0 / 3600.0)

            segment_lat = segment['geometry']['x']
            segment_lon = segment['geometry']['y']

            repeated_zone_multiplier = 1.0
            segment_penalty = 0.0
            segment_hit = False

            for cluster in verified_clusters:
                distance = haversine_distance_meters(
                    segment_lat,
                    segment_lon,
                    cluster.centroid_latitude,
                    cluster.centroid_longitude,
                )
                if distance > self.segment_influence_radius_m:
                    continue

                distance_factor = max(0.0, 1.0 - (distance / self.segment_influence_radius_m))
                confidence_factor = max(0.1, min(1.0, cluster.confidence_aggregate))
                density_factor = min(2.5, 1.0 + (cluster.reports_count / 4.0))

                severity_weight = self._cluster_severity_weight(cluster)
                risk_component = severity_weight * confidence_factor * density_factor * distance_factor

                if cluster.reports_count >= 5:
                    repeated_zone_multiplier = 1.3

                segment_penalty += self.base_penalty_seconds * risk_component
                segment_hit = True

                affected_coordinates.append({
                    'segment_id': segment_id,
                    'cluster_id': cluster.id,
                    'latitude': cluster.centroid_latitude,
                    'longitude': cluster.centroid_longitude,
                    'distance_meters': round(distance, 2),
                    'confidence': round(cluster.confidence_aggregate, 3),
                    'reports_count': cluster.reports_count,
                })

            segment_penalty *= repeated_zone_multiplier
            pothole_penalty_seconds += segment_penalty
            if segment_hit:
                pothole_warning_count += 1

        composite_score = eta_seconds + pothole_penalty_seconds
        route_risk_score = min(100.0, (pothole_penalty_seconds / max(eta_seconds, 1.0)) * 100.0)

        return {
            'path': path,
            'eta_seconds': round(eta_seconds, 2),
            'pothole_penalty_seconds': round(pothole_penalty_seconds, 2),
            'composite_score': round(composite_score, 2),
            'route_risk_score': round(route_risk_score, 2),
            'pothole_warning_count': pothole_warning_count,
            'affected_coordinates': affected_coordinates,
        }

    def _cluster_severity_weight(self, cluster):
        report_values = list(cluster.reports.values_list('severity', flat=True))
        if not report_values:
            return 0.8

        weights = {'minor': 0.5, 'moderate': 0.9, 'severe': 1.3}
        numeric = [weights.get(value, 0.8) for value in report_values]
        return sum(numeric) / len(numeric)

    def _format_route_result(self, score_dict):
        route_data = {
            'segments': score_dict['path'],
            'eta_seconds': score_dict['eta_seconds'],
            'pothole_penalty_seconds': score_dict['pothole_penalty_seconds'],
            'route_risk_score': score_dict['route_risk_score'],
            'pothole_warning_count': score_dict['pothole_warning_count'],
            'affected_coordinates': score_dict['affected_coordinates'],
            'composite_score': score_dict['composite_score'],
        }
        
        # Generate smart guidance for the route
        try:
            smart_guidance = self._guidance_service.generate_guidance(route_data=route_data)
            route_data['smart_guidance'] = smart_guidance
        except Exception as exc:
            logger.warning('Smart guidance generation failed: %s', exc)
            route_data['smart_guidance'] = None
            
        return route_data
