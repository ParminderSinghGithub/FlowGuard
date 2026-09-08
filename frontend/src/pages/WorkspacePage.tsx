import { FormEvent, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { APP_NAME } from '@/config';
import { clearToken } from '@/lib/auth';
import { getJson, postJson } from '@/lib/api';
import { ApiError } from '@/lib/errors';
import type {
  GeoPoint,
  NearbyPothole,
  PredictionHealthResponse,
  RouteOptimizationRequest,
  RouteOptimizationResponse,
  RouteSummary,
  RouteWarning,
  TrafficPredictionResponse,
} from '@/lib/contracts';
import { useGeoLocation } from '@/lib/geolocation';
import {
  getDemoDestination,
  promoteRecentLocation,
  readRecentLocations,
  resolveLocationSelection,
  searchLocations,
  toSelectedLocation,
  type LocationPoint,
  type SelectedLocation,
} from '@/lib/locations';
import { ErrorBanner } from '@/components/ErrorBanner';
import { LoadingPanel } from '@/components/LoadingPanel';
import { LocationPicker } from '@/components/LocationPicker';
import { SmartGuidance } from '@/components/SmartGuidance';
import { MapCanvas } from '@/map/MapCanvas';

const defaultDestination = getDemoDestination();

function createCurrentSelection(location: GeoPoint, simulated: boolean): SelectedLocation {
  return {
    id: simulated ? 'demo-current' : 'current',
    label: simulated ? 'Demo current location' : 'Current location',
    latitude: location.latitude,
    longitude: location.longitude,
    category: 'Current location',
    source: 'current',
  };
}

function haversineMeters(start: GeoPoint, endLatitude: number, endLongitude: number): number {
  const earthRadiusMeters = 6_371_000;
  const startLat = (start.latitude * Math.PI) / 180;
  const endLat = (endLatitude * Math.PI) / 180;
  const latDelta = ((endLatitude - start.latitude) * Math.PI) / 180;
  const lonDelta = ((endLongitude - start.longitude) * Math.PI) / 180;

  const a = Math.sin(latDelta / 2) ** 2 + Math.cos(startLat) * Math.cos(endLat) * Math.sin(lonDelta / 2) ** 2;
  return earthRadiusMeters * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function normalizeNearbyPotholes(payload: unknown, origin: GeoPoint | null): NearbyPothole[] {
  if (Array.isArray(payload)) {
    return payload
      .map((entry) => {
        if (!entry || typeof entry !== 'object') {
          return null;
        }

        const record = entry as {
          id?: number;
          cluster_id?: number;
          centroid_latitude?: number;
          centroid_longitude?: number;
          latitude?: number;
          longitude?: number;
          reports_count?: number;
          confidence_aggregate?: number;
          is_verified?: boolean;
        };

        const latitude = record.latitude ?? record.centroid_latitude;
        const longitude = record.longitude ?? record.centroid_longitude;

        if (typeof latitude !== 'number' || typeof longitude !== 'number') {
          return null;
        }

        return {
          cluster_id: record.cluster_id ?? record.id ?? 0,
          latitude,
          longitude,
          distance_meters: origin ? haversineMeters(origin, latitude, longitude) : 0,
          reports_count: record.reports_count ?? 0,
          confidence_aggregate: record.confidence_aggregate ?? 0,
          is_verified: record.is_verified ?? false,
        } satisfies NearbyPothole;
      })
      .filter((entry): entry is NearbyPothole => entry !== null);
  }

  if (payload && typeof payload === 'object' && 'potholes' in payload) {
    const maybePotholes = (payload as { potholes?: NearbyPothole[] }).potholes;
    return maybePotholes ?? [];
  }

  return [];
}

function normalizeRouteWarnings(payload: unknown): RouteWarning[] {
  if (Array.isArray(payload)) {
    return payload as RouteWarning[];
  }

  if (payload && typeof payload === 'object' && 'warnings' in payload) {
    const maybeWarnings = (payload as { warnings?: RouteWarning[] }).warnings;
    return maybeWarnings ?? [];
  }

  return [];
}

function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.round(seconds)} sec`;
  }

  return `${Math.round(seconds / 60)} min`;
}

export function WorkspacePage() {
  const navigate = useNavigate();
  const { location, loading: locationLoading, error: locationError, simulated: locationSimulated } = useGeoLocation(true);
  const [potholes, setPotholes] = useState<NearbyPothole[]>([]);
  const [routeWarnings, setRouteWarnings] = useState<RouteWarning[]>([]);
  const [traffic, setTraffic] = useState<TrafficPredictionResponse | null>(null);
  const [health, setHealth] = useState<PredictionHealthResponse | null>(null);
  const [routeResult, setRouteResult] = useState<RouteOptimizationResponse | null>(null);
  const [selectedRouteIndex, setSelectedRouteIndex] = useState(0);
  const [routeError, setRouteError] = useState<unknown>(null);
  const [mapError, setMapError] = useState<unknown>(null);
  const [routeLoading, setRouteLoading] = useState(false);
  const [mapLoading, setMapLoading] = useState(false);
  const [recentLocations, setRecentLocations] = useState<LocationPoint[]>(() => readRecentLocations());
  const [startQuery, setStartQuery] = useState(locationSimulated ? 'Demo current location' : 'Current location');
  const [destinationQuery, setDestinationQuery] = useState(defaultDestination.label);
  const [startSelection, setStartSelection] = useState<SelectedLocation | null>(null);
  const [destinationSelection, setDestinationSelection] = useState<SelectedLocation | null>(toSelectedLocation(defaultDestination));

  useEffect(() => {
    void getJson<PredictionHealthResponse>('/api/predict/health/')
      .then(setHealth)
      .catch(() => {
        setHealth({ model_ready: false, status: 'unavailable' });
      });

    void postJson<TrafficPredictionResponse>('/api/predict/')
      .then(setTraffic)
      .catch(() => {
        setTraffic(null);
      });
  }, []);

  useEffect(() => {
    if (!location) {
      return;
    }

    if (startSelection && startSelection.source !== 'current') {
      return;
    }

    const currentSelection = createCurrentSelection(location, locationSimulated);
    const needsUpdate =
      !startSelection ||
      startSelection.source !== currentSelection.source ||
      startSelection.label !== currentSelection.label ||
      startSelection.latitude !== currentSelection.latitude ||
      startSelection.longitude !== currentSelection.longitude;

    if (needsUpdate) {
      setStartSelection(currentSelection);
      setStartQuery(currentSelection.label);
    }
  }, [location, locationSimulated, startSelection]);

  useEffect(() => {
    if (!location) {
      return;
    }

    const timer = window.setTimeout(() => {
      setMapLoading(true);
      Promise.all([
        getJson<unknown>(`/api/potholes/nearby/?latitude=${location.latitude}&longitude=${location.longitude}&radius_meters=400`),
        getJson<unknown>(`/api/routes/warnings/?latitude=${location.latitude}&longitude=${location.longitude}&radius_meters=500`),
      ])
        .then(([nearby, warnings]) => {
          setPotholes(normalizeNearbyPotholes(nearby, location));
          setRouteWarnings(normalizeRouteWarnings(warnings));
          setMapError(null);
        })
        .catch((caughtError) => handleApiError(caughtError, setMapError))
        .finally(() => setMapLoading(false));
    }, 450);

    return () => window.clearTimeout(timer);
  }, [location]);

  useEffect(() => {
    if (!destinationSelection && defaultDestination) {
      setDestinationSelection(toSelectedLocation(defaultDestination));
      setDestinationQuery(defaultDestination.label);
    }
  }, [destinationSelection]);

  const routeOptions = useMemo(() => {
    if (!routeResult) {
      return [];
    }

    const seen = new Set<string>();
    return [routeResult.selected_route, ...routeResult.alternatives].filter((route) => {
      const key = `${route.eta_seconds}-${route.route_risk_score}-${route.pothole_warning_count}-${route.composite_score}`;
      if (seen.has(key)) {
        return false;
      }

      seen.add(key);
      return true;
    });
  }, [routeResult]);

  const routeSummary = routeOptions[selectedRouteIndex] ?? routeOptions[0] ?? null;

  const allSuggestions = useMemo(() => searchLocations(''), []);
  const startSuggestions = allSuggestions;
  const destinationSuggestions = allSuggestions;
  const recentStartSelections = useMemo(
    () => recentLocations.filter((locationPoint) => locationPoint.id !== destinationSelection?.id).slice(0, 4),
    [destinationSelection?.id, recentLocations],
  );
  const recentDestinationSelections = useMemo(
    () => recentLocations.filter((locationPoint) => locationPoint.id !== startSelection?.id).slice(0, 4),
    [recentLocations, startSelection?.id],
  );

  const routeStartPoint = startSelection
    ? {
        latitude: startSelection.latitude,
        longitude: startSelection.longitude,
      }
    : location;

  const routeDestinationPoint = destinationSelection
    ? {
        latitude: destinationSelection.latitude,
        longitude: destinationSelection.longitude,
      }
    : null;

  function promoteAndRemember(locationPoint: LocationPoint) {
    const nextRecent = promoteRecentLocation(locationPoint, recentLocations);
    setRecentLocations(nextRecent);
  }

  function handlePickStart(locationPoint: LocationPoint) {
    const selected = toSelectedLocation(locationPoint);
    setStartSelection(selected);
    setStartQuery(locationPoint.label);
    promoteAndRemember(locationPoint);
  }

  function handlePickDestination(locationPoint: LocationPoint) {
    const selected = toSelectedLocation(locationPoint);
    setDestinationSelection(selected);
    setDestinationQuery(locationPoint.label);
    promoteAndRemember(locationPoint);
  }

  function handleUseCurrentLocation() {
    if (!location) {
      return;
    }

    const currentSelection = createCurrentSelection(location, locationSimulated);
    setStartSelection(currentSelection);
    setStartQuery(currentSelection.label);
  }

  function handleSwapLocations() {
    const nextStart = destinationSelection;
    const nextDestination = startSelection;

    if (nextStart) {
      setStartSelection(nextStart);
      setStartQuery(nextStart.label);
    }

    if (nextDestination) {
      setDestinationSelection(nextDestination);
      setDestinationQuery(nextDestination.label);
    }
  }

  async function handleOptimize(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const resolvedStart = resolveLocationSelection(startQuery, startSelection);
    const resolvedDestination = resolveLocationSelection(destinationQuery, destinationSelection);

    if (!resolvedStart || !resolvedDestination) {
      setRouteError('Pick both locations from the searchable list before optimizing the route.');
      return;
    }

    setRouteLoading(true);
    setRouteError(null);

    const requestBody: RouteOptimizationRequest = {
      start_latitude: resolvedStart.latitude,
      start_longitude: resolvedStart.longitude,
      end_latitude: resolvedDestination.latitude,
      end_longitude: resolvedDestination.longitude,
      alternatives_count: 3,
      eta_tolerance_ratio: 1.15,
    };

    try {
      const optimized = await postJson<RouteOptimizationResponse>('/api/routes/optimize/', requestBody);
      setRouteResult(optimized);
      setSelectedRouteIndex(0);
    } catch (caughtError) {
      handleApiError(caughtError, setRouteError);
    } finally {
      setRouteLoading(false);
    }
  }

  function handleSignOut() {
    clearToken();
    navigate('/login', { replace: true });
  }

  function handleApiError(caughtError: unknown, setError: (error: unknown) => void) {
    if (caughtError instanceof ApiError && (caughtError.status === 401 || caughtError.status === 403)) {
      clearToken();
      navigate('/login', { replace: true });
      return;
    }

    setError(caughtError);
  }

  const routeStartSelection = resolveLocationSelection(startQuery, startSelection);
  const routeDestinationSelection = resolveLocationSelection(destinationQuery, destinationSelection);

  const riskLevelClass = useMemo(() => {
    if (!routeSummary) {
      return '';
    }
    if (routeSummary.route_risk_score >= 60) {
      return 'risk-high';
    }
    if (routeSummary.route_risk_score >= 25) {
      return 'risk-med';
    }
    return 'risk-low';
  }, [routeSummary]);

  const riskLabel = useMemo(() => {
    if (!routeSummary) {
      return 'Standby';
    }

    if (routeSummary.route_risk_score >= 60) {
      return 'High risk';
    }

    if (routeSummary.route_risk_score >= 25) {
      return 'Moderate risk';
    }

    return 'Low risk';
  }, [routeSummary]);

  const affectedCoordinates = routeSummary?.affected_coordinates ?? [];
  const alternativeRoutes = routeOptions;

  return (
    <main className="workspace">
      <header className="topbar">
        <div className="topbar__left">
          <div className="brand-icon-wrapper" aria-hidden="true">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              <path d="m9 12 2 2 4-4" />
            </svg>
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span className="eyebrow">{APP_NAME}</span>
              <span className="live-pill">
                <span className="live-dot"></span> Live Telemetry
              </span>
            </div>
            <h1>Intelligent Urban Navigation Grid</h1>
          </div>
        </div>
        <button className="button button-secondary" type="button" onClick={handleSignOut}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
            <polyline points="16 17 21 12 16 7" />
            <line x1="21" y1="12" x2="9" y2="12" />
          </svg>
          Sign out
        </button>
      </header>

      <section className="status-strip">
        <div className="status-card">
          <div className="status-card__info">
            <span className="status-label">Traffic Congestion</span>
            <strong>{traffic ? `${Math.round(traffic.prediction * 100)}% load` : 'Analyzing...'}</strong>
          </div>
          <div className="status-icon-pill traffic" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="6" y="2" width="12" height="20" rx="3" />
              <circle cx="12" cy="7" r="1.5" />
              <circle cx="12" cy="12" r="1.5" />
              <circle cx="12" cy="17" r="1.5" />
            </svg>
          </div>
        </div>

        <div className="status-card">
          <div className="status-card__info">
            <span className="status-label">Route Risk Status</span>
            <strong>{riskLabel}</strong>
          </div>
          <div className={`status-icon-pill ${riskLevelClass || 'risk-low'}`} aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
          </div>
        </div>

        <div className="status-card">
          <div className="status-card__info">
            <span className="status-label">Hazard Sensors</span>
            <strong>{potholes.length} nearby</strong>
          </div>
          <div className="status-icon-pill" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
              <line x1="12" y1="9" x2="12" y2="13" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
          </div>
        </div>

        <div className="status-card">
          <div className="status-card__info">
            <span className="status-label">ML Model</span>
            <strong>{health?.model_ready ? 'Online' : 'Operational'}</strong>
          </div>
          <div className="status-icon-pill risk-low" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
          </div>
        </div>
      </section>

      {locationError ? <ErrorBanner error={locationError} /> : null}
      {locationSimulated ? (
        <p className="inline-note">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="16" x2="12" y2="12" />
            <line x1="12" y1="8" x2="12.01" y2="8" />
          </svg>
          Using simulated Ludhiana urban coordinates because browser geolocation is restricted or unavailable.
        </p>
      ) : null}
      {routeError ? <ErrorBanner error={routeError} /> : null}
      {mapError ? <ErrorBanner error={mapError} /> : null}

      <section className="workspace-grid">
        <div className="map-panel">
          {locationLoading ? <LoadingPanel title="Acquiring GPS fix" description="Connecting to geolocation satellites..." /> : null}
          {mapLoading ? <LoadingPanel title="Querying Road Hazards" description="Scanning road telemetry in 500m radius..." /> : null}
          <MapCanvas
            location={location}
            potholes={potholes}
            route={routeSummary}
            alternatives={routeOptions}
            selectedRouteIndex={selectedRouteIndex}
            routeStart={routeStartPoint}
            routeEnd={routeDestinationPoint}
          />
        </div>

        <aside className="side-panel">
          <form className="route-form" onSubmit={handleOptimize}>
            <div className="route-form__header">
              <div>
                <h2>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="3 11 22 2 13 21 11 13 3 11" />
                  </svg>
                  Route Planner
                </h2>
                <p className="muted">Select origin and destination to compute hazard-minimized routes.</p>
              </div>
              <button className="button button-secondary button-inline" type="button" onClick={handleSwapLocations} title="Swap Start and Destination">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="m7 16 5 5 5-5" />
                  <path d="M12 21V9" />
                  <path d="m17 8-5-5-5 5" />
                  <path d="M12 3v12" />
                </svg>
                Swap
              </button>
            </div>

            <LocationPicker
              label="Start location"
              placeholder="Select start location"
              value={routeStartSelection}
              suggestions={startSuggestions}
              recentSelections={recentStartSelections}
              helperText="Use your current GPS coordinate or a saved landmark."
              onPick={handlePickStart}
              onUseCurrentLocation={handleUseCurrentLocation}
            />

            <LocationPicker
              label="Destination"
              placeholder="Select destination"
              value={routeDestinationSelection}
              suggestions={destinationSuggestions}
              recentSelections={recentDestinationSelections}
              helperText="Pick a target point from the urban location catalog."
              onPick={handlePickDestination}
            />

            <button className="button button-primary" type="submit" disabled={routeLoading || !routeStartSelection || !routeDestinationSelection}>
              {routeLoading ? (
                <>
                  <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px', borderTopColor: '#fff' }} />
                  Optimizing path...
                </>
              ) : (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
                  </svg>
                  Calculate Optimized Route
                </>
              )}
            </button>
          </form>

          <section className="summary-card">
            <h2>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <path d="m14 10-4 4" />
                <path d="m10 10 4 4" />
              </svg>
              Route Summary
            </h2>
            {routeSummary ? (
              <>
                <dl className="summary-list">
                  <div>
                    <dt>Risk score</dt>
                    <dd>{routeSummary.route_risk_score.toFixed(2)}</dd>
                  </div>
                  <div>
                    <dt>Est. Travel Time</dt>
                    <dd>{formatDuration(routeSummary.eta_seconds)}</dd>
                  </div>
                  <div>
                    <dt>Pothole warnings</dt>
                    <dd>{routeSummary.pothole_warning_count}</dd>
                  </div>
                  <div>
                    <dt>Hazard Penalty</dt>
                    <dd>{formatDuration(routeSummary.pothole_penalty_seconds)}</dd>
                  </div>
                </dl>

                <h3>Warnings on selected path</h3>
                <ul className="mini-list">
                  {affectedCoordinates.length > 0 ? (
                    affectedCoordinates.map((warning) => (
                      <li key={`${warning.cluster_id}-${warning.segment_id}`}>
                        <span>Hazard Cluster #{warning.cluster_id}</span>
                        <small style={{ color: 'var(--color-warning)', fontFamily: 'var(--font-mono)' }}>{warning.distance_meters.toFixed(0)}m away</small>
                      </li>
                    ))
                  ) : (
                    <li className="muted">No road warnings on the selected path. Clear roadway!</li>
                  )}
                </ul>
              </>
            ) : (
              <p className="muted">Select start and destination coordinates to trigger AI route optimization.</p>
            )}
          </section>

          <SmartGuidance 
            guidance={routeSummary?.smart_guidance || null} 
            isLoading={routeLoading} 
          />

          <section className="summary-card">
            <h2>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="6" cy="18" r="3" />
                <circle cx="18" cy="6" r="3" />
                <path d="M6 15V9a6 6 0 0 1 6-6h3" />
              </svg>
              Alternative Routes
            </h2>
            <ul className="mini-list">
              {alternativeRoutes.length > 0 ? (
                alternativeRoutes.map((alternative, index) => (
                  <li key={`${alternative.composite_score}-${index}`} style={{ padding: 0, border: 'none', background: 'transparent' }}>
                    <button
                      className={`route-option ${index === selectedRouteIndex ? 'route-option-active' : ''}`}
                      type="button"
                      onClick={() => setSelectedRouteIndex(index)}
                    >
                      <span>Route Option #{index + 1}</span>
                      <strong>{formatDuration(alternative.eta_seconds)}</strong>
                      <small>Risk: {alternative.route_risk_score.toFixed(1)}</small>
                    </button>
                  </li>
                ))
              ) : (
                <li className="muted">No alternative routes computed yet. Click calculate to generate.</li>
              )}
            </ul>
          </section>

          <section className="summary-card">
            <h2>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
              Nearby Pothole Clusters
            </h2>
            <ul className="mini-list">
              {potholes.length > 0 ? (
                potholes.map((pothole) => (
                  <li key={pothole.cluster_id}>
                    <span>Cluster #{pothole.cluster_id}</span>
                    <span style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                      <small style={{ color: 'var(--muted)' }}>{pothole.distance_meters.toFixed(0)}m</small>
                      <span className="badge" style={{ background: pothole.is_verified ? 'rgba(244,63,94,0.15)' : 'rgba(245,158,11,0.15)', color: pothole.is_verified ? '#fda4af' : '#fde68a' }}>
                        {pothole.reports_count} reports
                      </span>
                    </span>
                  </li>
                ))
              ) : (
                <li className="muted">No road hazards detected in the current sector. Safe travels!</li>
              )}
            </ul>
          </section>

          <section className="summary-card">
            <h2>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 8v4" />
                <path d="M12 16h.01" />
                <circle cx="12" cy="12" r="9" />
              </svg>
              Sector Road Warnings
            </h2>
            <ul className="mini-list">
              {routeWarnings.length > 0 ? (
                routeWarnings.map((warning: RouteWarning) => (
                  <li key={warning.cluster_id}>
                    <span>Cluster #{warning.cluster_id}</span>
                    <span style={{ color: 'var(--color-warning)', fontSize: '0.8rem' }}>{warning.warning}</span>
                  </li>
                ))
              ) : (
                <li className="muted">No sector road warnings registered.</li>
              )}
            </ul>
          </section>
        </aside>
      </section>
    </main>
  );
}
