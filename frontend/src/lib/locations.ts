import { MAP_CENTER } from '@/config';

export type LocationPoint = {
  id: string;
  label: string;
  latitude: number;
  longitude: number;
  category: string;
  aliases?: string[];
};

export type SelectedLocation = LocationPoint & {
  source: 'catalog' | 'current';
};

const RECENT_LOCATIONS_STORAGE_KEY = 'flowguard.recent.locations';

export const LOCATION_CATALOG: LocationPoint[] = [
  {
    id: 'clock-tower-chowk',
    label: 'Clock Tower Chowk',
    latitude: 30.9128,
    longitude: 75.8536,
    category: 'City landmark',
    aliases: ['Ghanta Ghar', 'Clock Tower'],
  },
  {
    id: 'sarabha-nagar-market',
    label: 'Sarabha Nagar Market',
    latitude: 30.9048,
    longitude: 75.8204,
    category: 'Shopping hub',
    aliases: ['Sarabha Nagar', 'Sarabha Market'],
  },
  {
    id: 'pau-gate',
    label: 'PAU Gate',
    latitude: 30.9001,
    longitude: 75.8087,
    category: 'Institution gate',
    aliases: ['Punjab Agricultural University Gate'],
  },
  {
    id: 'ferozepur-road-brs',
    label: 'Ferozepur Road at BRS Nagar',
    latitude: 30.8889,
    longitude: 75.7989,
    category: 'Road junction',
    aliases: ['BRS Nagar', 'Ferozepur Road'],
  },
  {
    id: 'model-town-extension',
    label: 'Model Town Extension',
    latitude: 30.9272,
    longitude: 75.8439,
    category: 'Residential area',
    aliases: ['Model Town'],
  },
  {
    id: 'pakhowal-road',
    label: 'Pakhowal Road',
    latitude: 30.8687,
    longitude: 75.7898,
    category: 'Arterial corridor',
    aliases: ['Pakhowal'],
  },
  {
    id: 'gill-road-industrial',
    label: 'Gill Road Industrial Area',
    latitude: 30.8951,
    longitude: 75.8674,
    category: 'Industrial belt',
    aliases: ['Gill Road'],
  },
  {
    id: 'ludhiana-railway-station',
    label: 'Ludhiana Railway Station',
    latitude: 30.9023,
    longitude: 75.8471,
    category: 'Transit hub',
    aliases: ['Railway Station'],
  },
  {
    id: 'ludhiana-bus-stand',
    label: 'Ludhiana Bus Stand',
    latitude: 30.9039,
    longitude: 75.8622,
    category: 'Transit hub',
    aliases: ['Bus Stand'],
  },
  {
    id: 'sidhwan-canal-road',
    label: 'Sidhwan Canal Road',
    latitude: 30.8786,
    longitude: 75.8203,
    category: 'Scenic connector',
    aliases: ['Canal Road'],
  },
  {
    id: 'dhandari-kalan',
    label: 'Dhandari Kalan',
    latitude: 30.8419,
    longitude: 75.8962,
    category: 'Logistics area',
    aliases: ['Dhandari'],
  },
  {
    id: 'haibowal-kalan',
    label: 'Haibowal Kalan',
    latitude: 30.9425,
    longitude: 75.8011,
    category: 'Residential area',
    aliases: ['Haibowal'],
  },
  {
    id: 'verka-milk-plant',
    label: 'Verka Milk Plant',
    latitude: 30.8874,
    longitude: 75.8391,
    category: 'Industry and logistics',
    aliases: ['Verka Plant'],
  },
  {
    id: 'ghumar-mandi',
    label: 'Ghumar Mandi',
    latitude: 30.9171,
    longitude: 75.8352,
    category: 'Shopping street',
    aliases: ['Ghumar'],
  },
];

export function toSelectedLocation(point: LocationPoint, source: SelectedLocation['source'] = 'catalog'): SelectedLocation {
  return { ...point, source };
}

export function locationMatchesQuery(location: LocationPoint, query: string): boolean {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) {
    return true;
  }

  const haystacks = [location.label, location.category, ...(location.aliases ?? [])].map((value) => value.toLowerCase());
  return haystacks.some((value) => value.includes(normalizedQuery));
}

export function searchLocations(query: string, limit = 6): LocationPoint[] {
  const normalizedQuery = query.trim().toLowerCase();
  const ranked = LOCATION_CATALOG
    .map((location) => {
      if (!normalizedQuery) {
        return { location, score: 0 };
      }

      const label = location.label.toLowerCase();
      const aliases = (location.aliases ?? []).map((alias) => alias.toLowerCase());
      let score = 0;

      if (label === normalizedQuery || aliases.includes(normalizedQuery)) {
        score += 100;
      }

      if (label.startsWith(normalizedQuery)) {
        score += 50;
      }

      if (aliases.some((alias) => alias.startsWith(normalizedQuery))) {
        score += 40;
      }

      if (label.includes(normalizedQuery)) {
        score += 20;
      }

      if ((location.aliases ?? []).some((alias) => alias.toLowerCase().includes(normalizedQuery))) {
        score += 15;
      }

      if (location.category.toLowerCase().includes(normalizedQuery)) {
        score += 8;
      }

      return { location, score };
    })
    .filter(({ location, score }) => score > 0 || !normalizedQuery)
    .sort((left, right) => right.score - left.score || left.location.label.localeCompare(right.location.label));

  return ranked.slice(0, limit).map(({ location }) => location);
}

export function resolveLocationSelection(query: string, selected: LocationPoint | null): LocationPoint | null {
  if (selected) {
    return selected;
  }

  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) {
    return null;
  }

  return (
    LOCATION_CATALOG.find((location) => {
      if (location.label.toLowerCase() === normalizedQuery) {
        return true;
      }

      return (location.aliases ?? []).some((alias) => alias.toLowerCase() === normalizedQuery);
    }) ?? searchLocations(query, 1)[0] ?? null
  );
}

export function getDemoDestination(): LocationPoint {
  return LOCATION_CATALOG.find((location) => location.id === 'ferozepur-road-brs') ?? LOCATION_CATALOG[0] ?? {
    id: 'demo-destination',
    label: 'Demo destination',
    latitude: MAP_CENTER[0],
    longitude: MAP_CENTER[1],
    category: 'Demo',
  };
}

export function readRecentLocations(): LocationPoint[] {
  if (typeof window === 'undefined') {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(RECENT_LOCATIONS_STORAGE_KEY);
    if (!raw) {
      return [];
    }

    const parsed = JSON.parse(raw) as LocationPoint[];
    return parsed.filter((location) => typeof location?.label === 'string');
  } catch {
    return [];
  }
}

export function storeRecentLocations(locations: LocationPoint[]): void {
  if (typeof window === 'undefined') {
    return;
  }

  window.localStorage.setItem(RECENT_LOCATIONS_STORAGE_KEY, JSON.stringify(locations.slice(0, 5)));
}

export function promoteRecentLocation(location: LocationPoint, current: LocationPoint[] = readRecentLocations()): LocationPoint[] {
  const filtered = current.filter((item) => item.id !== location.id);
  const next = [location, ...filtered].slice(0, 5);
  storeRecentLocations(next);
  return next;
}
