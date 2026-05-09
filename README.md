# FlowGuard

FlowGuard is a Python-first traffic intelligence platform built on Django and Django REST Framework.

## Overview

The backend currently focuses on:

- Real-time traffic ingestion from TomTom
- Congestion prediction using a TensorFlow Lite model
- Route and traffic data APIs via DRF
- Pothole reporting and notification entities
- Scheduled ingestion using Celery tasks

## Architecture

1. Data ingestion
   - `app.tasks.fetch_ludhiana_traffic`
   - `app.management.commands.fetch_tomtom_data`
2. Prediction
   - TFLite model in `server/app/tflite_model/models/`
   - Inference endpoint in `app/views.py` (`/api/predict/`)
3. Route and traffic APIs
   - DRF router endpoints in `app/urls.py`
4. Persistence
   - SQLite database (`server/db.sqlite3`)

## Tech Stack

- Python
- Django + Django REST Framework
- SQLite
- Celery (optional runtime worker/beat)
- TensorFlow Lite + scikit-learn

## Status

This repository is under active backend cleanup and completion.

## License

Distributed under the MIT License. See `LICENSE` for details.

