export class ApiError extends Error {
  status: number;
  code: string;
  details?: unknown;
  retryAfterSeconds?: number;

  constructor(message: string, init: { status: number; code: string; details?: unknown; retryAfterSeconds?: number }) {
    super(message);
    this.name = 'ApiError';
    this.status = init.status;
    this.code = init.code;
    this.details = init.details;
    this.retryAfterSeconds = init.retryAfterSeconds;
  }
}

export function formatErrorDetails(details: unknown): string | null {
  if (!details) {
    return null;
  }

  if (typeof details === 'string') {
    return details;
  }

  if (Array.isArray(details)) {
    return details.map(formatErrorDetails).filter(Boolean).join(' ');
  }

  if (typeof details === 'object') {
    return Object.entries(details)
      .map(([field, value]) => {
        const text = formatErrorDetails(value);
        return text ? `${humanizeFieldName(field)}: ${text}` : null;
      })
      .filter(Boolean)
      .join(' ');
  }

  return String(details);
}

export function humanizeFieldName(field: string): string {
  return field.replace(/_/g, ' ').replace(/^\w/, (character: string) => character.toUpperCase());
}

export function isRateLimitError(error: unknown): error is ApiError {
  return error instanceof ApiError && error.status === 429;
}
