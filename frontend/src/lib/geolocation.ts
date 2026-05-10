import { useEffect, useState } from 'react';
import { MAP_CENTER } from '@/config';
import type { GeoPoint } from '@/lib/contracts';

type LocationState = {
  location: GeoPoint | null;
  loading: boolean;
  error: string | null;
  simulated: boolean;
};

export function useGeoLocation(autoStart = true): LocationState {
  const [state, setState] = useState<LocationState>({
    location: null,
    loading: autoStart,
    error: null,
    simulated: false,
  });

  useEffect(() => {
    if (!autoStart) {
      return;
    }

    if (!navigator.geolocation) {
      setState({
        location: { latitude: MAP_CENTER[0], longitude: MAP_CENTER[1] },
        loading: false,
        error: 'Geolocation is not available in this browser. Using a demo location instead.',
        simulated: true,
      });
      return;
    }

    const watchId = navigator.geolocation.watchPosition(
      (position) => {
        setState({
          location: {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
          },
          loading: false,
          error: null,
          simulated: false,
        });
      },
      (error) => {
        setState({
          location: { latitude: MAP_CENTER[0], longitude: MAP_CENTER[1] },
          loading: false,
          error: error.message === 'User denied Geolocation'
            ? 'Location permission was denied. Using a demo location instead.'
            : error.message,
          simulated: true,
        });
      },
      {
        enableHighAccuracy: true,
        maximumAge: 10_000,
        timeout: 15_000,
      },
    );

    return () => navigator.geolocation.clearWatch(watchId);
  }, [autoStart]);

  return state;
}
