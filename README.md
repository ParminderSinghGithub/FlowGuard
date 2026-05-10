# FlowGuard

FlowGuard is a full-stack pothole-aware traffic intelligence MVP for safer urban route planning. It combines a Django REST backend, React map workspace, pothole detection pipeline, and route-risk scoring to help drivers compare routes around road hazards in Ludhiana.

## Live Demo

[ Add deployment link here ]

## Demo GIF

[ Add demo GIF here ]

## Screenshots

[ Add screenshots here ]

## Features

- Token-based signup, login, logout, and protected workspace routing.
- Map-first React interface with Leaflet and OpenStreetMap tiles.
- Name-based location search with recent locations, current-location shortcut, route swapping, and seeded Ludhiana places.
- Nearby pothole rendering with verified cluster markers.
- Pothole-aware route optimization with risk scores, warnings, ETA impact, and selectable alternatives.
- Sensor ingestion pipeline for accelerometer-based pothole detection.
- Manual pothole reporting and cluster verification APIs.
- Traffic prediction health endpoint with graceful degraded mode when ML assets are unavailable.
- Celery worker and beat support for periodic traffic ingestion, cluster verification, and cleanup jobs.
- SQLite persistence and WhiteNoise-backed Django static handling for deployment preparation.

## Architecture

FlowGuard is split into a Django API server and a Vite React frontend.

```text
frontend/       React + Vite + TypeScript map workspace
server/         Django project, DRF API, Celery app, SQLite database
server/app/     Domain models, serializers, views, services, tasks, tests
```

The frontend calls relative `/api/...` endpoints in development, and Vite proxies them to Django at `http://127.0.0.1:8000`. In production, serve the frontend build from your web server and proxy `/api` to Django.

## Tech Stack

- Backend: Python, Django, Django REST Framework, DRF token auth
- Database: SQLite
- Background jobs: Celery, django-celery-beat, django-celery-results
- Frontend: React, Vite, TypeScript
- Maps: Leaflet, React Leaflet, OpenStreetMap
- ML/data: TensorFlow CPU/TFLite, NumPy, pandas, scikit-learn, joblib
- Static files: WhiteNoise

## Pothole Detection

FlowGuard accepts authenticated sensor readings with latitude, longitude, device ID, and accelerometer Z-axis data. The pothole service builds a per-device baseline, detects significant spikes, debounces duplicate reports, clusters nearby reports, and marks clusters verified when enough reports and distinct devices agree.

The relevant APIs are:

- `POST /api/potholes/sensor/`
- `POST /api/potholes/report/`
- `GET /api/potholes/nearby/`
- `POST /api/potholes/verify-clusters/`
- `GET /api/routes/warnings/`

## Route Optimization

Route optimization combines route geometry, pothole cluster proximity, warning counts, ETA penalty, and risk scoring. The backend returns a selected route plus alternatives; the frontend lets users switch between options and see the active risk summary on the map.

The main routing APIs are:

- `POST /api/routes/optimize/`
- `POST /api/routes/alternatives/`
- `POST /api/routes/risk-analysis/`

## Authentication

FlowGuard uses DRF token authentication.

- Signup: `POST /api/auth/register/`
- Login: `POST /api/auth/token/`
- Auth header: `Authorization: Token <token>`

Signup requires:

```json
{
  "username": "demo-user",
  "password": "StrongPass123!",
  "password_confirmation": "StrongPass123!"
}
```

The frontend stores the token in `localStorage`, attaches it to protected requests, clears stale tokens on `401` or `403`, and redirects unauthenticated users to login.

## Background Jobs

Celery is optional for local demos but supported for production-like operation.

Periodic jobs include:

- Fetch Ludhiana traffic every 2 minutes.
- Verify pothole clusters every 5 minutes.
- Clean stale sensor points every 6 hours.
- Clean read notifications daily.
- Clean inactive unverified clusters daily.
- Refresh route cache every 15 minutes.

If the broker is unavailable, cluster verification falls back to synchronous execution for the API flow that needs it.

## Setup

Prerequisites:

- Python 3.12
- Node.js and npm
- A Python virtual environment at `.venv`

Create `.env` from `.env.example` and set a real `SECRET_KEY`.

```powershell
Copy-Item .env.example .env
```

Create the virtual environment if it is not already present, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm install
```

Run migrations:

```powershell
cd server
..\.venv\Scripts\python.exe manage.py migrate
```

## Local Development

Local development is HTTP-only:

- Frontend: `http://127.0.0.1:5174`
- Backend: `http://127.0.0.1:8000`
- API through Vite: `http://127.0.0.1:5174/api/...`

Start Django:

```powershell
cd server
..\.venv\Scripts\python.exe manage.py runserver 127.0.0.1:8000
```

Start Vite:

```powershell
cd frontend
npm run dev
```

Build the frontend:

```powershell
cd frontend
npm run build
```

Run backend tests:

```powershell
cd server
..\.venv\Scripts\python.exe manage.py test app
```

Collect static files:

```powershell
cd server
..\.venv\Scripts\python.exe manage.py collectstatic --noinput
```

## Environment Variables

Important local defaults:

```env
FLOWGUARD_ENV=development
DEBUG=true
ALLOWED_HOSTS=127.0.0.1,localhost
SECURE_SSL_REDIRECT=false
SESSION_COOKIE_SECURE=false
CSRF_COOKIE_SECURE=false
SECURE_HSTS_SECONDS=0
SECURE_PROXY_SSL_HEADER=false
VITE_API_BASE_URL=/api
```

Production HTTPS defaults:

```env
FLOWGUARD_ENV=production
DEBUG=false
ALLOWED_HOSTS=your-domain.example
SECURE_SSL_REDIRECT=true
SESSION_COOKIE_SECURE=true
CSRF_COOKIE_SECURE=true
SECURE_HSTS_SECONDS=31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS=true
SECURE_HSTS_PRELOAD=true
SECURE_PROXY_SSL_HEADER=true
VITE_API_BASE_URL=/api
```

Optional variables include:

- `TOMTOM_API_KEY`
- `CELERY_BROKER_URL`
- `CELERY_RESULT_BACKEND`
- `SQLITE_DB_PATH`
- throttle and pothole tuning values from `.env.example`

Do not commit `.env`; it is intentionally ignored.

## Running Celery

Worker:

```powershell
cd server
..\.venv\Scripts\celery.exe -A server worker -l info --pool=solo
```

Beat:

```powershell
cd server
..\.venv\Scripts\celery.exe -A server beat -l info
```

## Deployment Notes

- Keep `FLOWGUARD_ENV=production` and `DEBUG=false` in production.
- Use HTTPS at the edge and enable the secure cookie/HSTS variables.
- Serve `frontend/dist` from your web server or hosting platform.
- Proxy `/api` to the Django app.
- Run `migrate` and `collectstatic --noinput` during release.
- SQLite is retained for this MVP; move to PostgreSQL for higher-concurrency production use.
- Django admin static files are served through WhiteNoise after `collectstatic`.

## Project Structure

```text
FlowGuard/
├── frontend/
│   ├── src/components/      Reusable UI components
│   ├── src/lib/             API, auth, errors, geolocation, locations
│   ├── src/map/             Leaflet map canvas
│   └── src/pages/           Login, signup, workspace
├── server/
│   ├── app/models.py        Domain models
│   ├── app/serializers.py   API validation and serialization
│   ├── app/views.py         DRF views and API endpoints
│   ├── app/services/        Pothole and route intelligence services
│   ├── app/tasks.py         Celery tasks
│   └── server/settings.py   Django configuration
├── .env.example
├── requirements.txt
└── README.md
```

## Known Limitations

- SQLite is suitable for the MVP/demo profile, not high-write production traffic.
- Traffic prediction can report `degraded` if bundled TFLite assets require unsupported Flex ops in the local interpreter.
- The location catalog is seeded for Ludhiana and configured in `frontend/src/lib/locations.ts`.
- Browser geolocation falls back to a demo Ludhiana location when permission is denied or unavailable.
- Celery improves scheduled processing, but core demo flows remain usable without a running worker.

## Future Improvements

- Deploy a live hosted demo.
- Add screenshots and the final product demo GIF.
- Move production persistence to PostgreSQL/PostGIS.
- Add richer map clustering for very large pothole datasets.
- Add CI for backend tests and frontend builds.
- Add refresh-token or session expiry UX if auth requirements expand.
- Expand location catalogs beyond Ludhiana.

## License

Distributed under the MIT License. See `LICENSE` for details.
