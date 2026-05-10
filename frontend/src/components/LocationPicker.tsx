import { useEffect, useMemo, useRef, useState } from 'react';
import type { LocationPoint } from '@/lib/locations';

type Props = {
  label: string;
  placeholder: string;
  query: string;
  value: LocationPoint | null;
  suggestions: LocationPoint[];
  recentSelections: LocationPoint[];
  helperText?: string;
  onQueryChange: (value: string) => void;
  onPick: (location: LocationPoint) => void;
  onUseCurrentLocation?: () => void;
};

export function LocationPicker({
  label,
  placeholder,
  query,
  value,
  suggestions,
  recentSelections,
  helperText,
  onQueryChange,
  onPick,
  onUseCurrentLocation,
}: Props) {
  const [open, setOpen] = useState(false);
  const blurTimer = useRef<number | null>(null);

  const filteredSuggestions = useMemo(() => suggestions.slice(0, 6), [suggestions]);

  useEffect(() => {
    return () => {
      if (blurTimer.current) {
        window.clearTimeout(blurTimer.current);
      }
    };
  }, []);

  function handleFocus() {
    setOpen(true);
  }

  function handleBlur() {
    blurTimer.current = window.setTimeout(() => setOpen(false), 120);
  }

  function handleSelect(location: LocationPoint) {
    onPick(location);
    onQueryChange(location.label);
    setOpen(false);
  }

  return (
    <div className="location-picker">
      <label>
        <span className="location-picker__label">{label}</span>
        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          onFocus={handleFocus}
          onBlur={handleBlur}
          placeholder={placeholder}
          autoComplete="off"
          spellCheck={false}
        />
      </label>

      {helperText ? <p className="form-hint">{helperText}</p> : null}

      {value ? (
        <p className="location-picker__selection">
          Selected: <strong>{value.label}</strong>
        </p>
      ) : null}

      <div className="location-picker__actions">
        {onUseCurrentLocation ? (
          <button className="button button-secondary button-inline" type="button" onClick={onUseCurrentLocation}>
            Use current location
          </button>
        ) : null}
      </div>

      {open ? (
        <div className="location-picker__menu" role="listbox">
          {filteredSuggestions.length > 0 ? (
            <div className="location-picker__section">
              <p className="location-picker__section-title">Suggestions</p>
              {filteredSuggestions.map((location) => (
                <button key={location.id} type="button" className="location-picker__item" onMouseDown={(event) => event.preventDefault()} onClick={() => handleSelect(location)}>
                  <strong>{location.label}</strong>
                  <span>{location.category}</span>
                </button>
              ))}
            </div>
          ) : null}

          {recentSelections.length > 0 ? (
            <div className="location-picker__section">
              <p className="location-picker__section-title">Recent</p>
              <div className="location-picker__chips">
                {recentSelections.map((location) => (
                  <button key={location.id} type="button" className="location-chip" onMouseDown={(event) => event.preventDefault()} onClick={() => handleSelect(location)}>
                    {location.label}
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          {filteredSuggestions.length === 0 && recentSelections.length === 0 ? <p className="location-picker__empty">No matching locations yet.</p> : null}
        </div>
      ) : null}
    </div>
  );
}
