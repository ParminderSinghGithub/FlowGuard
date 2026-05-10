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
      <div>
        <strong>Request failed</strong>
        <p>{message}</p>
        {details ? <p className="notice-detail">{details}</p> : null}
        {code ? <span className="notice-meta">Code: {code}</span> : null}
      </div>
      {onRetry ? (
        <button className="button button-secondary" type="button" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  );
}
