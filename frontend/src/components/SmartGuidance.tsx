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
        <h2>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
          </svg>
          Smart Driving Guidance
        </h2>
        <div className="loading-placeholder">
          <div className="spinner" style={{ margin: '0 auto 12px' }} />
          <p className="muted">Analyzing real-time road conditions & computing guidance...</p>
        </div>
      </section>
    );
  }

  if (!guidance) {
    return (
      <section className="summary-card">
        <h2>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 16v-4" />
            <path d="M12 8h.01" />
          </svg>
          Smart Driving Guidance
        </h2>
        <p className="muted">Optimize a route to generate real-time AI driving advisory and speed recommendations.</p>
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
        <h2>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
          </svg>
          Smart Driving Guidance
        </h2>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {guidance.fallback_mode && (
            <span className="badge badge-warning">Limited Telemetry</span>
          )}
          <div className="confidence-indicator">
            <span className="confidence-label">AI Confidence</span>
            <span className="confidence-value">
              {Math.round(guidance.confidence_score * 100)}%
            </span>
          </div>
        </div>
      </div>

      <div className="guidance-content">
        {/* Speed Recommendation */}
        <div className="guidance-section">
          <h3>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: 'inline', marginRight: '6px', verticalAlign: 'middle' }}>
              <circle cx="12" cy="12" r="10" />
              <path d="m14 14-4-4" />
              <path d="M12 6v2" />
            </svg>
            Recommended Velocity
          </h3>
          <div className="speed-range">
            <span className="speed-value">
              {guidance.recommended_speed_range.recommended_speed_kmh.toFixed(0)} km/h
            </span>
            <span className="speed-range-detail">
              Optimal window: {guidance.recommended_speed_range.min_speed_kmh.toFixed(0)} – {guidance.recommended_speed_range.max_speed_kmh.toFixed(0)} km/h
            </span>
          </div>
        </div>

        {/* Traffic Conditions */}
        <div className="guidance-section">
          <h3>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: 'inline', marginRight: '6px', verticalAlign: 'middle' }}>
              <rect x="6" y="2" width="12" height="20" rx="3" />
              <circle cx="12" cy="7" r="1.5" />
              <circle cx="12" cy="12" r="1.5" />
              <circle cx="12" cy="17" r="1.5" />
            </svg>
            Traffic Flow Density
          </h3>
          <div className="condition-indicator">
            <span 
              className="condition-level" 
              style={{ color: getCongestionColor(guidance.congestion_level.level) }}
            >
              ● {guidance.congestion_level.level.toUpperCase()} CONGESTION
            </span>
            <span className="condition-description">
              {guidance.congestion_level.description}
            </span>
          </div>
        </div>

        {/* Road Quality */}
        <div className="guidance-section">
          <h3>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: 'inline', marginRight: '6px', verticalAlign: 'middle' }}>
              <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            </svg>
            Surface Condition
          </h3>
          <div className="condition-indicator">
            <span 
              className="condition-level" 
              style={{ color: getRoadQualityColor(guidance.road_quality_warning.quality_level) }}
            >
              ● {guidance.road_quality_warning.quality_level.toUpperCase()} QUALITY
            </span>
            <span className="condition-description">
              {guidance.road_quality_warning.description}
            </span>
          </div>
          {guidance.road_quality_warning.has_immediate_hazards && (
            <div className="hazard-alert">
              ⚠️ Immediate hazards detected on this path
            </div>
          )}
        </div>

        {/* Time Pressure */}
        <div className="guidance-section">
          <h3>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: 'inline', marginRight: '6px', verticalAlign: 'middle' }}>
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
            Time Impact & Delay
          </h3>
          <div className="condition-indicator">
            <span 
              className="condition-level" 
              style={{ color: getPressureColor(guidance.eta_pressure.level) }}
            >
              ● {guidance.eta_pressure.level.toUpperCase()} DELAY RISK
            </span>
            <span className="condition-description">
              {guidance.eta_pressure.description}
            </span>
          </div>
          {guidance.eta_pressure.penalty_minutes > 0 && (
            <div className="time-impact">
              ⏱ +{guidance.eta_pressure.penalty_minutes.toFixed(1)} min road condition delay
            </div>
          )}
        </div>

        {/* Driving Recommendations */}
        <div className="guidance-section">
          <h3>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: 'inline', marginRight: '6px', verticalAlign: 'middle' }}>
              <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
            </svg>
            Advisory Directives
          </h3>
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
