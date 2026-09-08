import { useMemo, useState } from 'react';
import type { LocationPoint } from '@/lib/locations';

type Props = {
  label: string;
  placeholder: string;
  value: LocationPoint | null;
  suggestions: LocationPoint[];
  recentSelections: LocationPoint[];
  helperText?: string;
  onPick: (location: LocationPoint) => void;
  onUseCurrentLocation?: () => void;
};

export function LocationPicker({
  label,
  placeholder,
  value,
  suggestions,
  recentSelections,
  helperText,
  onPick,
  onUseCurrentLocation,
}: Props) {
  const [isOpen, setIsOpen] = useState(false);

  const allOptions = useMemo(() => {
    const options: Array<LocationPoint | { id: string; label: string; category: string; isCurrentLocation: boolean }> = [];
    
    // Add current location option if available
    if (onUseCurrentLocation) {
      options.push({
        id: 'current-location',
        label: 'Use current GPS location',
        category: 'Live GPS',
        isCurrentLocation: true,
        latitude: 0,
        longitude: 0,
      });
    }
    
    // Add recent selections
    if (recentSelections.length > 0) {
      options.push(...recentSelections);
    }
    
    // Add suggestions
    options.push(...suggestions);
    
    return options;
  }, [suggestions, recentSelections, onUseCurrentLocation]);

  function handleSelect(location: LocationPoint | { id: string; label: string; category: string; isCurrentLocation: boolean }) {
    if ('isCurrentLocation' in location && location.isCurrentLocation && onUseCurrentLocation) {
      onUseCurrentLocation();
    } else if ('latitude' in location && 'longitude' in location) {
      onPick(location as LocationPoint);
    }
    setIsOpen(false);
  }

  return (
    <div className="location-picker">
      <label>
        <span className="location-picker__label">{label}</span>
        <div className="location-picker__dropdown">
          <button
            type="button"
            className="location-picker__trigger"
            onClick={() => setIsOpen(!isOpen)}
            aria-expanded={isOpen}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', overflow: 'hidden' }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--primary-glow)', flexShrink: 0 }}>
                <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z" />
                <circle cx="12" cy="10" r="3" />
              </svg>
              <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                {value ? value.label : placeholder}
              </span>
            </div>
            <span
              className="location-picker__arrow"
              style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)' }}
            >
              ▼
            </span>
          </button>
          
          {isOpen && (
            <div className="location-picker__menu">
              <div className="location-picker__scrollable">
                {allOptions.map((location) => (
                  <button
                    key={location.id}
                    type="button"
                    className="location-picker__item"
                    onClick={() => handleSelect(location)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--muted)', flexShrink: 0 }}>
                        <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
                        <circle cx="12" cy="10" r="3" />
                      </svg>
                      <strong>{location.label}</strong>
                    </div>
                    <span>{location.category}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </label>

      {helperText ? <p className="form-hint">{helperText}</p> : null}
    </div>
  );
}
