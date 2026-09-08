import { CircleMarker, MapContainer, Polyline, TileLayer, Tooltip, useMap } from 'react-leaflet';
import type { LatLngBoundsExpression } from 'leaflet';
import { useEffect, useMemo } from 'react';
import { MAP_CENTER, MAP_ZOOM } from '@/config';
import type { GeoPoint, NearbyPothole, RouteAffectedCoordinate, RouteSummary } from '@/lib/contracts';

type Props = {
  location: GeoPoint | null;
  potholes: NearbyPothole[];
  route: RouteSummary | null;
  alternatives: RouteSummary[];
  selectedRouteIndex: number;
  routeStart: GeoPoint | null;
  routeEnd: GeoPoint | null;
};

function useFitBounds(points: Array<[number, number]>) {
  const map = useMap();
  const boundsKey = points.map((point) => point.join(',')).join('|');

  useEffect(() => {
    if (points.length === 0) {
      return;
    }

    const bounds: LatLngBoundsExpression = points;
    map.fitBounds(bounds, { padding: [36, 36], maxZoom: 15, animate: true, duration: 0.5 });
  }, [boundsKey, map, points]);
}

function useMapResize(points: Array<[number, number]>) {
  const map = useMap();
  const boundsKey = points.map((point) => point.join(',')).join('|');

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      map.invalidateSize();
    });

    const handleResize = () => map.invalidateSize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener('resize', handleResize);
    };
  }, [boundsKey, map]);
}

function riskColor(riskScore: number): string {
  if (riskScore >= 60) {
    return '#f43f5e';
  }

  if (riskScore >= 25) {
    return '#f59e0b';
  }

  return '#10b981';
}

function routePoints(routeStart: GeoPoint | null, routeEnd: GeoPoint | null, route: RouteSummary | null): Array<[number, number]> {
  const points: Array<[number, number]> = [];

  if (routeStart) {
    points.push([routeStart.latitude, routeStart.longitude]);
  }

  const affectedCoordinates = route?.affected_coordinates ?? [];

  if (affectedCoordinates.length > 0) {
    affectedCoordinates.forEach((coordinate: RouteAffectedCoordinate) => {
      points.push([coordinate.latitude, coordinate.longitude]);
    });
  }

  if (routeEnd) {
    points.push([routeEnd.latitude, routeEnd.longitude]);
  }

  return points;
}

export function MapCanvas({ location, potholes, route, alternatives, selectedRouteIndex, routeStart, routeEnd }: Props) {
  const allPoints = useMemo(() => {
    const points: Array<[number, number]> = [];

    if (location) {
      points.push([location.latitude, location.longitude]);
    }

    potholes.forEach((pothole) => {
      points.push([pothole.latitude, pothole.longitude]);
    });

    routePoints(routeStart, routeEnd, route).forEach((point) => points.push(point));

    return points;
  }, [location, potholes, route, routeStart, routeEnd]);

  const selectedRouteLine = routeStart && routeEnd ? ([
    [routeStart.latitude, routeStart.longitude],
    [routeEnd.latitude, routeEnd.longitude],
  ] as Array<[number, number]>) : null;

  const alternativeLines = useMemo(() => {
    if (!routeStart || !routeEnd) {
      return [];
    }

    return alternatives.map((alternative, index) => {
      const offset = (index - selectedRouteIndex) * 0.0018;
      return {
        route: alternative,
        index,
        positions: [
          [routeStart.latitude + offset, routeStart.longitude - offset],
          [routeEnd.latitude + offset, routeEnd.longitude - offset],
        ] as Array<[number, number]>,
      };
    });
  }, [alternatives, routeEnd, routeStart, selectedRouteIndex]);

  return (
    <div className="map-shell">
      <MapContainer center={MAP_CENTER} zoom={MAP_ZOOM} className="map-canvas" scrollWheelZoom>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <MapBounds points={allPoints} />

        {/* User GPS Location */}
        {location ? (
          <CircleMarker
            center={[location.latitude, location.longitude]}
            radius={8}
            pathOptions={{
              color: '#14b8a6',
              fillColor: '#06b6d4',
              fillOpacity: 0.95,
              weight: 3,
            }}
          >
            <Tooltip direction="top" offset={[0, -10]} opacity={1} permanent>
              GPS Location
            </Tooltip>
          </CircleMarker>
        ) : null}

        {/* Start Point Marker (if different from GPS or explicitly picked) */}
        {routeStart && (!location || Math.abs(routeStart.latitude - location.latitude) > 0.0001 || Math.abs(routeStart.longitude - location.longitude) > 0.0001) ? (
          <CircleMarker
            center={[routeStart.latitude, routeStart.longitude]}
            radius={7}
            pathOptions={{
              color: '#10b981',
              fillColor: '#34d399',
              fillOpacity: 0.9,
              weight: 2,
            }}
          >
            <Tooltip direction="top" offset={[0, -8]} opacity={1}>
              Origin Point
            </Tooltip>
          </CircleMarker>
        ) : null}

        {/* Destination Marker */}
        {routeEnd ? (
          <CircleMarker
            center={[routeEnd.latitude, routeEnd.longitude]}
            radius={8}
            pathOptions={{
              color: '#6366f1',
              fillColor: '#818cf8',
              fillOpacity: 0.9,
              weight: 2,
            }}
          >
            <Tooltip direction="top" offset={[0, -8]} opacity={1}>
              Destination
            </Tooltip>
          </CircleMarker>
        ) : null}

        {/* Pothole Road Hazards */}
        {potholes.map((pothole) => (
          <CircleMarker
            key={pothole.cluster_id}
            center={[pothole.latitude, pothole.longitude]}
            radius={Math.max(6, Math.min(16, pothole.reports_count * 2 + 4))}
            pathOptions={{
              color: pothole.is_verified ? '#f43f5e' : '#f59e0b',
              fillColor: pothole.is_verified ? '#f43f5e' : '#f59e0b',
              fillOpacity: 0.45,
              weight: 2,
            }}
          >
            <Tooltip direction="top" offset={[0, -8]} opacity={1}>
              Hazard #{pothole.cluster_id}: {pothole.reports_count} reports ({Math.round(pothole.confidence_aggregate * 100)}% conf.)
            </Tooltip>
          </CircleMarker>
        ))}

        {/* Alternative Routes */}
        {alternativeLines.map((line) => line.index === selectedRouteIndex ? null : (
          <Polyline
            key={`alternative-${line.index}`}
            positions={line.positions}
            pathOptions={{
              color: riskColor(line.route.route_risk_score),
              weight: 3,
              opacity: 0.45,
              dashArray: '5 7',
            }}
          />
        ))}

        {/* Selected Route */}
        {selectedRouteLine ? (
          <Polyline
            positions={selectedRouteLine}
            pathOptions={{
              color: riskColor(route?.route_risk_score ?? 0),
              weight: 5,
              opacity: 0.9,
              dashArray: '8 6',
            }}
          />
        ) : null}

        {/* Affected Coordinates / Hazard Hotspots on Route */}
        {(route?.affected_coordinates ?? []).map((point) => (
          <CircleMarker
            key={`${point.cluster_id}-${point.segment_id}`}
            center={[point.latitude, point.longitude]}
            radius={9}
            pathOptions={{
              color: '#e11d48',
              fillColor: '#fb7185',
              fillOpacity: 0.75,
              weight: 2,
            }}
          >
            <Tooltip direction="top" offset={[0, -8]} opacity={1}>
              Route Warning #{point.cluster_id} ({point.distance_meters.toFixed(0)}m)
            </Tooltip>
          </CircleMarker>
        ))}
      </MapContainer>
    </div>
  );
}

function MapBounds({ points }: { points: Array<[number, number]> }) {
  useFitBounds(points);
  useMapResize(points);
  return null;
}
