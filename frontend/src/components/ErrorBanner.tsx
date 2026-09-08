import { ApiError, formatErrorDetails } from '@/lib/errors';

type Props = {
  error: unknown;
  onRetry?: () => void;
};

export function ErrorBanner({ error, onRetry }: Props) {
  const message = error instanceof ApiError ? error.message : typeof error === 'string' ? error : error instanceof Error ? error.message : 'Something went wrong.';
  const code = error instanceof ApiError ? error.code : undefined;
  const details = error instanceof ApiError ? formatErrorDetails(error.details) : null;

  return (
    <div className="notice notice-error" role="alert">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--color-danger)', flexShrink: 0, marginTop: '2px' }}>
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>
      <div style={{ flex: 1 }}>
        <strong>Request Failed</strong>
        <p>{message}</p>
        {details ? <p className="notice-detail">{details}</p> : null}
        {code ? <span className="notice-meta">Code: {code}</span> : null}
      </div>
      {onRetry ? (
        <button className="button button-secondary button-inline" type="button" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  );
}
