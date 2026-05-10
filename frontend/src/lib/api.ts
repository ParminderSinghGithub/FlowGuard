import { apiUrl } from '@/config';
import { clearToken, readToken } from '@/lib/auth';
import { ApiError } from '@/lib/errors';

type RequestOptions = RequestInit & { retryCount?: number };

export async function requestJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const token = readToken();
  const headers = new Headers(options.headers ?? {});
  headers.set('Accept', 'application/json');

  if (options.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  if (token) {
    headers.set('Authorization', `Token ${token}`);
  }

  const response = await fetch(apiUrl(path), {
    ...options,
    headers,
  });

  if (!response.ok) {
    const error = await parseApiError(response);
    if (error.status === 401 || error.status === 403) {
      clearToken();
    }
    throw error;
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export async function requestWithRetry<T>(path: string, options: RequestOptions = {}): Promise<T> {
  try {
    return await requestJson<T>(path, options);
  } catch (error) {
    if (!(error instanceof ApiError)) {
      throw error;
    }

    const shouldRetry = error.status === 429 || error.status >= 500;
    const retryCount = options.retryCount ?? 0;

    if (!shouldRetry || retryCount >= 1) {
      throw error;
    }

    const delaySeconds = error.retryAfterSeconds ?? 1;
    await new Promise((resolve) => window.setTimeout(resolve, delaySeconds * 1000));
    return requestWithRetry<T>(path, { ...options, retryCount: retryCount + 1 });
  }
}

export async function getJson<T>(path: string): Promise<T> {
  return requestWithRetry<T>(path, { method: 'GET' });
}

export async function postJson<T>(path: string, body?: unknown): Promise<T> {
  return requestWithRetry<T>(path, {
    method: 'POST',
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

async function parseApiError(response: Response): Promise<ApiError> {
  let payload: unknown;

  try {
    payload = await response.json();
  } catch {
    payload = undefined;
  }

  const errorObject = payload && typeof payload === 'object' && 'error' in payload ? (payload as { error?: { code?: string; message?: string; details?: unknown } }).error : undefined;

  return new ApiError(errorObject?.message ?? response.statusText ?? 'Request failed', {
    status: response.status,
    code: errorObject?.code ?? 'request_failed',
    details: errorObject?.details,
    retryAfterSeconds: Number.parseInt(response.headers.get('Retry-After') ?? '', 10) || undefined,
  });
}
