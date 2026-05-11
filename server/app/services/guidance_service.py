import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.models import PotholeCluster, TrafficData, CongestionPrediction
from app.services.geo import haversine_distance_meters

logger = logging.getLogger(__name__)


class SmartGuidanceService:
    """Lightweight smart driving guidance service using existing ML and route data."""

    def __init__(self):
        self.speed_limits = {
            'urban': 40.0,      # km/h
            'highway': 80.0,    # km/h
            'residential': 25.0 # km/h
        }
        
        self.congestion_thresholds = {
            'low': 0.3,
            'moderate': 0.6,
            'high': 0.8
        }

    def generate_guidance(self, *, route_data: Dict, traffic_data: Optional[List] = None, 
                          congestion_predictions: Optional[List] = None) -> Dict:
        """
        Generate smart driving guidance for a route.
        
        Args:
            route_data: Route optimization result from RouteIntelligenceService
            traffic_data: Current traffic data for route segments
            congestion_predictions: ML-based congestion predictions
            
        Returns:
            Dictionary containing guidance information
        """
        try:
            guidance = {
                'recommended_speed_range': self._calculate_speed_range(route_data),
                'congestion_level': self._assess_congestion_level(route_data, traffic_data, congestion_predictions),
                'road_quality_warning': self._assess_road_quality(route_data),
                'driving_recommendations': [],
                'confidence_score': 0.0,
                'eta_pressure': self._calculate_eta_pressure(route_data)
            }
            
            # Generate contextual recommendations
            guidance['driving_recommendations'] = self._generate_recommendations(guidance, route_data)
            
            # Calculate overall confidence
            guidance['confidence_score'] = self._calculate_confidence_score(guidance, route_data)
            
            return guidance
            
        except Exception as exc:
            logger.exception('Error generating smart guidance: %s', exc)
            return self._get_fallback_guidance()

    def _calculate_speed_range(self, route_data: Dict) -> Dict[str, float]:
        """Calculate recommended speed range based on route conditions."""
        base_speed = 35.0  # Default urban speed in km/h
        
        # Adjust for pothole risk
        risk_score = route_data.get('route_risk_score', 0)
        pothole_warnings = route_data.get('pothole_warning_count', 0)
        
        # Lower speed recommendation for high-risk routes
        risk_adjustment = max(0.5, 1.0 - (risk_score / 100.0) * 0.4)
        
        # Additional adjustment for pothole density
        pothole_adjustment = max(0.7, 1.0 - (pothole_warnings / 10.0) * 0.3)
        
        recommended_speed = base_speed * risk_adjustment * pothole_adjustment
        
        # Create speed range (±5 km/h around recommended)
        min_speed = max(15.0, recommended_speed - 5)
        max_speed = min(60.0, recommended_speed + 5)
        
        return {
            'min_speed_kmh': round(min_speed, 1),
            'max_speed_kmh': round(max_speed, 1),
            'recommended_speed_kmh': round(recommended_speed, 1)
        }

    def _assess_congestion_level(self, route_data: Dict, traffic_data: Optional[List], 
                                congestion_predictions: Optional[List]) -> Dict:
        """Assess traffic congestion level for the route."""
        # Use ML predictions if available, otherwise use historical patterns
        if congestion_predictions:
            avg_congestion = np.mean([pred.get('congestion_index', 0.5) for pred in congestion_predictions])
        elif traffic_data:
            # Calculate from current traffic speed ratios
            speed_ratios = [data.get('speed_ratio', 1.0) for data in traffic_data]
            avg_congestion = 1.0 - (np.mean(speed_ratios) if speed_ratios else 0.5)
        else:
            # Use time-based estimation
            current_hour = datetime.now().hour
            if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
                avg_congestion = 0.7  # Rush hours
            elif 10 <= current_hour <= 16:
                avg_congestion = 0.4  # Daytime
            else:
                avg_congestion = 0.2  # Off-peak
        
        # Determine congestion level
        if avg_congestion <= self.congestion_thresholds['low']:
            level = 'low'
            description = 'Light traffic expected'
        elif avg_congestion <= self.congestion_thresholds['moderate']:
            level = 'moderate'
            description = 'Moderate congestion predicted'
        elif avg_congestion <= self.congestion_thresholds['high']:
            level = 'high'
            description = 'Heavy congestion expected'
        else:
            level = 'severe'
            description = 'Severe congestion - consider alternative route'
        
        return {
            'level': level,
            'description': description,
            'congestion_index': round(avg_congestion, 3)
        }

    def _assess_road_quality(self, route_data: Dict) -> Dict:
        """Assess road quality based on pothole data."""
        risk_score = route_data.get('route_risk_score', 0)
        pothole_warnings = route_data.get('pothole_warning_count', 0)
        affected_coordinates = route_data.get('affected_coordinates', [])
        
        # Determine road quality level
        if risk_score <= 20:
            quality_level = 'good'
            description = 'Road conditions appear good'
        elif risk_score <= 40:
            quality_level = 'fair'
            description = 'Some road irregularities expected'
        elif risk_score <= 60:
            quality_level = 'poor'
            description = 'Poor road conditions - drive carefully'
        else:
            quality_level = 'hazardous'
            description = 'Hazardous road conditions - extreme caution required'
        
        # Identify high-risk segments
        high_risk_segments = [
            coord for coord in affected_coordinates 
            if coord.get('confidence', 0) > 0.7 and coord.get('reports_count', 0) >= 3
        ]
        
        return {
            'quality_level': quality_level,
            'description': description,
            'risk_score': round(risk_score, 2),
            'pothole_warnings': pothole_warnings,
            'high_risk_segments_count': len(high_risk_segments),
            'has_immediate_hazards': len(high_risk_segments) > 0
        }

    def _calculate_eta_pressure(self, route_data: Dict) -> Dict:
        """Calculate time pressure for the route."""
        eta_seconds = route_data.get('eta_seconds', 0)
        penalty_seconds = route_data.get('pothole_penalty_seconds', 0)
        
        # Calculate pressure as percentage of total time that's penalty
        if eta_seconds > 0:
            pressure_ratio = penalty_seconds / eta_seconds
        else:
            pressure_ratio = 0
        
        if pressure_ratio <= 0.1:
            pressure_level = 'low'
            description = 'Comfortable travel time'
        elif pressure_ratio <= 0.25:
            pressure_level = 'moderate'
            description = 'Some time pressure due to road conditions'
        else:
            pressure_level = 'high'
            description = 'Significant time impact from road conditions'
        
        return {
            'level': pressure_level,
            'description': description,
            'pressure_ratio': round(pressure_ratio, 3),
            'base_eta_minutes': round(eta_seconds / 60, 1),
            'penalty_minutes': round(penalty_seconds / 60, 1)
        }

    def _generate_recommendations(self, guidance: Dict, route_data: Dict) -> List[str]:
        """Generate contextual driving recommendations."""
        recommendations = []
        
        # Speed recommendations
        speed_range = guidance['recommended_speed_range']
        rec_speed = speed_range['recommended_speed_kmh']
        recommendations.append(f"Recommended speed: {rec_speed:.0f}–{speed_range['max_speed_kmh']:.0f} km/h")
        
        # Congestion recommendations
        congestion = guidance['congestion_level']
        if congestion['level'] in ['moderate', 'high', 'severe']:
            recommendations.append(congestion['description'])
            if congestion['level'] == 'severe':
                recommendations.append("Consider alternative route if available")
        
        # Road quality recommendations
        road_quality = guidance['road_quality_warning']
        if road_quality['quality_level'] in ['poor', 'hazardous']:
            recommendations.append("Reduce speed near uneven road sections")
            if road_quality['has_immediate_hazards']:
                recommendations.append("Watch for potholes and road damage")
        
        # ETA pressure recommendations
        eta_pressure = guidance['eta_pressure']
        if eta_pressure['level'] == 'high':
            recommendations.append("Allow extra time for route delays")
        
        # General safety recommendations
        if route_data.get('pothole_warning_count', 0) > 5:
            recommendations.append("Maintain safe following distance")
        
        # Time-based recommendations
        current_hour = datetime.now().hour
        if 6 <= current_hour <= 8:
            recommendations.append("Morning commute - expect increased traffic")
        elif 17 <= current_hour <= 19:
            recommendations.append("Evening rush hour - plan for delays")
        
        return recommendations[:4]  # Limit to 4 most relevant recommendations

    def _calculate_confidence_score(self, guidance: Dict, route_data: Dict) -> float:
        """Calculate overall confidence in the guidance."""
        confidence_factors = []
        
        # Route data completeness
        if route_data.get('route_risk_score') is not None:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
        
        # Pothole data quality
        affected_coords = route_data.get('affected_coordinates', [])
        if affected_coords:
            avg_confidence = np.mean([coord.get('confidence', 0.5) for coord in affected_coords])
            confidence_factors.append(avg_confidence)
        else:
            confidence_factors.append(0.3)
        
        # Congestion prediction confidence
        congestion = guidance['congestion_level']
        if congestion['congestion_index'] > 0:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # Overall confidence (weighted average)
        return round(np.mean(confidence_factors), 3)

    def _get_fallback_guidance(self) -> Dict:
        """Return safe fallback guidance when service is unavailable."""
        return {
            'recommended_speed_range': {
                'min_speed_kmh': 25.0,
                'max_speed_kmh': 45.0,
                'recommended_speed_kmh': 35.0
            },
            'congestion_level': {
                'level': 'moderate',
                'description': 'Traffic conditions unknown - drive cautiously',
                'congestion_index': 0.5
            },
            'road_quality_warning': {
                'quality_level': 'fair',
                'description': 'Road conditions unknown - stay alert',
                'risk_score': 30.0,
                'pothole_warnings': 0,
                'high_risk_segments_count': 0,
                'has_immediate_hazards': False
            },
            'driving_recommendations': [
                'Drive at safe speeds',
                'Stay alert for road conditions',
                'Allow extra travel time'
            ],
            'confidence_score': 0.3,
            'eta_pressure': {
                'level': 'moderate',
                'description': 'Travel time estimates unavailable',
                'pressure_ratio': 0.0,
                'base_eta_minutes': 0,
                'penalty_minutes': 0
            },
            'fallback_mode': True
        }
