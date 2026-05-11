import type { RouteOptimizationResponse } from '@/lib/contracts';

interface SmartGuidanceData {
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
}

interface SmartGuidanceProps {
  guidance: SmartGuidanceData | null;
  isLoading?: boolean;
}

export function SmartGuidance({ guidance, isLoading }: SmartGuidanceProps) {
  if (isLoading) {
    return (
      <section className="summary-card">
        <h2>Smart Driving Guidance</h2>
        <div className="loading-placeholder">
          <p className="muted">Analyzing route conditions...</p>
        </div>
      </section>
    );
  }

  if (!guidance) {
    return (
      <section className="summary-card">
        <h2>Smart Driving Guidance</h2>
        <p className="muted">Route optimization required to generate guidance.</p>
      </section>
    );
  }

  const getCongestionColor = (level: string) => {
    switch (level) {
      case 'low': return 'var(--color-success)';
      case 'moderate': return 'var(--color-warning)';
      case 'high': return 'var(--color-danger)';
      case 'severe': return 'var(--color-danger)';
      default: return 'var(--color-text-muted)';
    }
  };

  const getRoadQualityColor = (level: string) => {
    switch (level) {
      case 'good': return 'var(--color-success)';
      case 'fair': return 'var(--color-warning)';
      case 'poor': return 'var(--color-danger)';
      case 'hazardous': return 'var(--color-danger)';
      default: return 'var(--color-text-muted)';
    }
  };

  const getPressureColor = (level: string) => {
    switch (level) {
      case 'low': return 'var(--color-success)';
      case 'moderate': return 'var(--color-warning)';
      case 'high': return 'var(--color-danger)';
      default: return 'var(--color-text-muted)';
    }
  };

  return (
    <section className="summary-card">
      <div className="guidance-header">
        <h2>Smart Driving Guidance</h2>
        {guidance.fallback_mode && (
          <span className="badge badge-warning">Limited Data</span>
        )}
        <div className="confidence-indicator">
          <span className="confidence-label">Confidence:</span>
          <span className="confidence-value">
            {Math.round(guidance.confidence_score * 100)}%
          </span>
        </div>
      </div>

      <div className="guidance-content">
        {/* Speed Recommendation */}
        <div className="guidance-section">
          <h3>Recommended Speed</h3>
          <div className="speed-range">
            <span className="speed-value">
              {guidance.recommended_speed_range.recommended_speed_kmh.toFixed(0)} km/h
            </span>
            <span className="speed-range-detail">
              {guidance.recommended_speed_range.min_speed_kmh.toFixed(0)}–{guidance.recommended_speed_range.max_speed_kmh.toFixed(0)} km/h
            </span>
          </div>
        </div>

        {/* Traffic Conditions */}
        <div className="guidance-section">
          <h3>Traffic Conditions</h3>
          <div className="condition-indicator">
            <span 
              className="condition-level" 
              style={{ color: getCongestionColor(guidance.congestion_level.level) }}
            >
              {guidance.congestion_level.level.charAt(0).toUpperCase() + guidance.congestion_level.level.slice(1)}
            </span>
            <span className="condition-description">
              {guidance.congestion_level.description}
            </span>
          </div>
        </div>

        {/* Road Quality */}
        <div className="guidance-section">
          <h3>Road Quality</h3>
          <div className="condition-indicator">
            <span 
              className="condition-level" 
              style={{ color: getRoadQualityColor(guidance.road_quality_warning.quality_level) }}
            >
              {guidance.road_quality_warning.quality_level.charAt(0).toUpperCase() + guidance.road_quality_warning.quality_level.slice(1)}
            </span>
            <span className="condition-description">
              {guidance.road_quality_warning.description}
            </span>
          </div>
          {guidance.road_quality_warning.has_immediate_hazards && (
            <div className="hazard-alert">
              ⚠️ Immediate hazards detected
            </div>
          )}
        </div>

        {/* Time Pressure */}
        <div className="guidance-section">
          <h3>Time Impact</h3>
          <div className="condition-indicator">
            <span 
              className="condition-level" 
              style={{ color: getPressureColor(guidance.eta_pressure.level) }}
            >
              {guidance.eta_pressure.level.charAt(0).toUpperCase() + guidance.eta_pressure.level.slice(1)}
            </span>
            <span className="condition-description">
              {guidance.eta_pressure.description}
            </span>
          </div>
          {guidance.eta_pressure.penalty_minutes > 0 && (
            <div className="time-impact">
              +{guidance.eta_pressure.penalty_minutes.toFixed(1)} min due to road conditions
            </div>
          )}
        </div>

        {/* Driving Recommendations */}
        <div className="guidance-section">
          <h3>Recommendations</h3>
          <ul className="recommendations-list">
            {guidance.driving_recommendations.map((recommendation, index) => (
              <li key={index} className="recommendation-item">
                {recommendation}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
