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
        label: 'Use current location',
        category: 'GPS',
        isCurrentLocation: true,
        latitude: 0,
        longitude: 0
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
          >
            {value ? value.label : placeholder}
            <span className="location-picker__arrow">▼</span>
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
                    <strong>{location.label}</strong>
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
