export type GeoPoint = {
  latitude: number;
  longitude: number;
};

export type AuthTokenResponse = {
  token: string;
};

export type NearbyPothole = {
  cluster_id: number;
  latitude: number;
  longitude: number;
  distance_meters: number;
  reports_count: number;
  confidence_aggregate: number;
  is_verified: boolean;
  severity_levels?: string[];
};

export type NearbyPotholesResponse = {
  potholes: NearbyPothole[];
  count: number;
  search_radius_meters: number;
};

export type RouteAffectedCoordinate = {
  segment_id: number | string;
  cluster_id: number;
  latitude: number;
  longitude: number;
  distance_meters: number;
  confidence: number;
  reports_count: number;
};

export type RouteSummary = {
  segments: Array<number | string>;
  eta_seconds: number;
  pothole_penalty_seconds: number;
  route_risk_score: number;
  pothole_warning_count: number;
  affected_coordinates: RouteAffectedCoordinate[];
  composite_score: number;
  smart_guidance?: {
    recommended_speed_range: {
      min_speed_kmh: number;
      max_speed_kmh: number;
      recommended_speed_kmh: number;
    };
    congestion_level: {
      level: 'low' | 'moderate' | 'high' | 'severe';
      description: string;
      congestion_index: number;
    };
    road_quality_warning: {
      quality_level: 'good' | 'fair' | 'poor' | 'hazardous';
      description: string;
      risk_score: number;
      pothole_warnings: number;
      high_risk_segments_count: number;
      has_immediate_hazards: boolean;
    };
    driving_recommendations: string[];
    confidence_score: number;
    eta_pressure: {
      level: 'low' | 'moderate' | 'high';
      description: string;
      pressure_ratio: number;
      base_eta_minutes: number;
      penalty_minutes: number;
    };
    fallback_mode?: boolean;
  };
};

export type RouteOptimizationRequest = {
  start_latitude: number;
  start_longitude: number;
  end_latitude: number;
  end_longitude: number;
  departure_time?: string;
  eta_tolerance_ratio?: number;
  alternatives_count?: number;
};

export type RouteOptimizationResponse = {
  selected_route: RouteSummary;
  alternatives: RouteSummary[];
  graph_mode: string;
  fallback_mode: boolean;
};

export type RouteWarning = {
  cluster_id: number;
  latitude: number;
  longitude: number;
  reports_count: number;
  confidence_aggregate: number;
  warning: string;
};

export type RouteWarningsResponse = {
  warnings: RouteWarning[];
  warning_count: number;
  search_radius_meters: number;
};

export type PredictionHealthResponse = {
  model_ready: boolean;
  status: string;
  error?: string;
  message?: string;
};

export type TrafficPredictionResponse = {
  prediction: number;
  prediction_confidence: number;
  model_status: string;
  hotspots: Array<{ lat: number; lon: number }>;
};
