import { TOKEN_STORAGE_KEY, apiUrl } from '@/config';
import type { AuthTokenResponse } from '@/lib/contracts';
import { ApiError, formatErrorDetails } from '@/lib/errors';

export function readToken(): string | null {
  if (typeof window === 'undefined') {
    return null;
  }

  return window.localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function storeToken(token: string): void {
  window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
}

export function clearToken(): void {
  window.localStorage.removeItem(TOKEN_STORAGE_KEY);
}

export async function login(username: string, password: string): Promise<string> {
  const response = await fetch(apiUrl('/api/auth/token/'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ username, password }),
  });

  if (!response.ok) {
    throw await parseApiError(response);
  }

  const data = (await response.json()) as AuthTokenResponse;
  if (!data.token) {
    throw new ApiError('Token was not returned by the server.', { status: 500, code: 'invalid_auth_response' });
  }

  storeToken(data.token);
  return data.token;
}

export async function signup(username: string, password: string, passwordConfirmation: string): Promise<string> {
  const response = await fetch(apiUrl('/api/auth/register/'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ username, password, password_confirmation: passwordConfirmation }),
  });

  if (!response.ok) {
    throw await parseApiError(response);
  }

  const data = (await response.json()) as AuthTokenResponse;
  if (!data.token) {
    throw new ApiError('Token was not returned by the server.', { status: 500, code: 'invalid_auth_response' });
  }

  storeToken(data.token);
  return data.token;
}

async function parseApiError(response: Response): Promise<ApiError> {
  let payload: unknown;

  try {
    payload = await response.json();
  } catch {
    payload = undefined;
  }

  const errorObject = payload && typeof payload === 'object' && 'error' in payload ? (payload as { error?: { code?: string; message?: string; details?: unknown } }).error : undefined;
  const legacyDetails = payload && typeof payload === 'object' && 'errors' in payload ? (payload as { errors?: unknown }).errors : undefined;
  const details = errorObject?.details ?? legacyDetails;
  const detailText = formatErrorDetails(details);
  const message = errorObject?.message ?? detailText ?? response.statusText ?? 'Request failed';

  return new ApiError(message, {
    status: response.status,
    code: errorObject?.code ?? (response.status === 400 ? 'validation_error' : 'request_failed'),
    details,
    retryAfterSeconds: Number.parseInt(response.headers.get('Retry-After') ?? '', 10) || undefined,
  });
}
