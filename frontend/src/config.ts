export const APP_NAME = 'FlowGuard';
export const API_BASE_URL = normalizeApiBaseUrl(import.meta.env.VITE_API_BASE_URL);
export const TOKEN_STORAGE_KEY = 'flowguard.auth.token';
export const MAP_CENTER: [number, number] = [30.9005, 75.8573];
export const MAP_ZOOM = 14;

function normalizeApiBaseUrl(value: unknown): string {
  if (typeof value !== 'string' || value.trim() === '') {
    return '/api';
  }

  const baseUrl = value.trim().replace(/\/+$/, '');

  if (import.meta.env.DEV && /^https:\/\/(127\.0\.0\.1|localhost):8000(\/api)?$/i.test(baseUrl)) {
    const httpBaseUrl = baseUrl.replace(/^https:/i, 'http:');
    return /\/api$/i.test(httpBaseUrl) ? httpBaseUrl : `${httpBaseUrl}/api`;
  }

  if (import.meta.env.DEV && /^http:\/\/(127\.0\.0\.1|localhost):8000$/i.test(baseUrl)) {
    return `${baseUrl}/api`;
  }

  return baseUrl;
}

export function apiUrl(path: string): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${normalizedPath}`;
}
