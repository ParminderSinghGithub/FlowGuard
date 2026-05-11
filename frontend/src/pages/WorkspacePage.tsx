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
    void getJson<PredictionHealthResponse>('/predict/health/')
      .then(setHealth)
      .catch(() => {
        setHealth({ model_ready: false, status: 'unavailable' });
      });

    void postJson<TrafficPredictionResponse>('/predict/')
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
        getJson<unknown>(`/potholes/nearby/?latitude=${location.latitude}&longitude=${location.longitude}&radius_meters=400`),
        getJson<unknown>(`/routes/warnings/?latitude=${location.latitude}&longitude=${location.longitude}&radius_meters=500`),
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

  const allSuggestions = useMemo(() => searchLocations(''), []); // Get all locations with empty query
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
      const optimized = await postJson<RouteOptimizationResponse>('/routes/optimize/', requestBody);
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

  const riskLabel = useMemo(() => {
    if (!routeSummary) {
      return 'No route selected yet.';
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
        <div>
          <p className="eyebrow">{APP_NAME}</p>
          <h1>Intelligent Route Planner</h1>
        </div>
        <button className="button button-secondary" type="button" onClick={handleSignOut}>
          Sign out
        </button>
      </header>

      <section className="status-strip">
        <div className="status-card">
          <span className="status-label">Traffic</span>
          <strong>{traffic ? `${Math.round(traffic.prediction * 100)}% congestion` : 'loading'}</strong>
        </div>
        <div className="status-card">
          <span className="status-label">Route risk</span>
          <strong>{riskLabel}</strong>
        </div>
      </section>

      {locationError ? <ErrorBanner error={locationError} /> : null}
      {locationSimulated ? <p className="inline-note">Using the demo Ludhiana location because browser geolocation is unavailable.</p> : null}
      {routeError ? <ErrorBanner error={routeError} /> : null}
      {mapError ? <ErrorBanner error={mapError} /> : null}

      <section className="workspace-grid">
        <div className="map-panel">
          {locationLoading ? <LoadingPanel title="Detecting location" /> : null}
          {mapLoading ? <LoadingPanel title="Loading potholes" /> : null}
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
                <h2>Plan route</h2>
                <p className="muted">Choose named locations and compare route risk.</p>
              </div>
              <button className="button button-secondary button-inline" type="button" onClick={handleSwapLocations}>
                Swap
              </button>
            </div>

            <LocationPicker
              label="Start location"
              placeholder="Select start location"
              value={routeStartSelection}
              suggestions={startSuggestions}
              recentSelections={recentStartSelections}
              helperText="Use a saved place or current location."
              onPick={handlePickStart}
              onUseCurrentLocation={handleUseCurrentLocation}
            />

            <LocationPicker
              label="Destination"
              placeholder="Select destination"
              value={routeDestinationSelection}
              suggestions={destinationSuggestions}
              recentSelections={recentDestinationSelections}
              helperText="Pick a destination from the location catalog."
              onPick={handlePickDestination}
            />

            <button className="button button-primary" type="submit" disabled={routeLoading || !routeStartSelection || !routeDestinationSelection}>
              {routeLoading ? 'Optimizing route...' : 'Optimize route'}
            </button>
          </form>

          <section className="summary-card">
            <h2>Route summary</h2>
            {routeSummary ? (
              <>
                <dl className="summary-list">
                  <div>
                    <dt>Risk score</dt>
                    <dd>{routeSummary.route_risk_score.toFixed(2)}</dd>
                  </div>
                  <div>
                    <dt>ETA</dt>
                    <dd>{formatDuration(routeSummary.eta_seconds)}</dd>
                  </div>
                  <div>
                    <dt>Pothole warnings</dt>
                    <dd>{routeSummary.pothole_warning_count}</dd>
                  </div>
                  <div>
                    <dt>Penalty</dt>
                    <dd>{formatDuration(routeSummary.pothole_penalty_seconds)}</dd>
                  </div>
                </dl>

                <h3>Warnings on route</h3>
                <ul className="mini-list">
                  {affectedCoordinates.length > 0 ? affectedCoordinates.map((warning) => (
                    <li key={`${warning.cluster_id}-${warning.segment_id}`}>
                      Cluster {warning.cluster_id} - {warning.distance_meters.toFixed(0)}m away
                    </li>
                  )) : (
                    <li className="muted">No route warnings on the selected path.</li>
                  )}
                </ul>
              </>
            ) : (
              <p className="muted">Select locations to begin intelligent route planning.</p>
            )}
          </section>

          <SmartGuidance 
            guidance={routeSummary?.smart_guidance || null} 
            isLoading={routeLoading} 
          />

          <section className="summary-card">
            <h2>Alternative routes</h2>
            <ul className="mini-list">
              {alternativeRoutes.length > 0 ? (
                alternativeRoutes.map((alternative, index) => (
                  <li key={`${alternative.composite_score}-${index}`}>
                    <button
                      className={`route-option ${index === selectedRouteIndex ? 'route-option-active' : ''}`}
                      type="button"
                      onClick={() => setSelectedRouteIndex(index)}
                    >
                      <span>Option {index + 1}</span>
                      <strong>{formatDuration(alternative.eta_seconds)}</strong>
                      <small>risk {alternative.route_risk_score.toFixed(1)}</small>
                    </button>
                  </li>
                ))
              ) : (
                <li className="muted">No alternative routes available. Try different locations.</li>
              )}
            </ul>
          </section>

          <section className="summary-card">
            <h2>Nearby potholes</h2>
            <ul className="mini-list">
              {potholes.length > 0 ? (
                potholes.map((pothole) => (
                  <li key={pothole.cluster_id}>
                    #{pothole.cluster_id} - {pothole.distance_meters.toFixed(0)}m - {pothole.reports_count} reports
                  </li>
                ))
              ) : (
                <li className="muted">No road hazards detected in this area. Safe travels!</li>
              )}
            </ul>
          </section>

          <section className="summary-card">
            <h2>Route warnings</h2>
            <ul className="mini-list">
              {routeWarnings.length > 0 ? (
                routeWarnings.map((warning: RouteWarning) => (
                  <li key={warning.cluster_id}>
                    #{warning.cluster_id} - {warning.warning}
                  </li>
                ))
              ) : (
                <li className="muted">No route warnings loaded yet.</li>
              )}
            </ul>
          </section>
        </aside>
      </section>
    </main>
  );
}
