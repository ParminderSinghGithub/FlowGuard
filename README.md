# FlowGuard

_An in-progress traffic congestion prediction and route optimization system for Ludhiana, leveraging real-time data, machine learning, and edge deployment._

## Live Demo

Explore the interactive Streamlit dashboard showcasing real-time traffic visualization and route suggestions:

[📊 View Live Demo](https://parmindersinghgithub-flowguard-serverapptflite-modelmain-sy0tzv.streamlit.app/#fe339b6c)
![Demo](demo.gif)
---

## Overview

FlowGuard is designed to:

- Collect and process real-time traffic data across Ludhiana’s key hotspots
- Predict future congestion levels and bottleneck risks using a spatiotemporal LSTM model
- Optimize routes dynamically, accounting for upcoming merges and predicted traffic shifts
- Provide interactive visualizations and statistics via a Streamlit interface
- Offer speed recommendations to help users avoid heavy congestion

---

## System Architecture

1. **Data Ingestion & Processing**  
   - Fetches data from TomTom Traffic API at 2‑minute intervals  
   - Normalizes speed and computes metrics such as free‑flow ratio and confidence scores

2. **Feature Engineering**  
   - Builds temporal sliding windows of traffic features per road segment  
   - Derives speed‑trend and spatial context matrices for model input

3. **Machine Learning Model**  
   - Spatiotemporal LSTM with multi‑head attention for sequence and spatial data  
   - Dual‑task outputs: speed prediction (MSE loss) and bottleneck risk (binary cross‑entropy loss)  
   - Converted to TensorFlow Lite for low‑latency edge inference

4. **Route Optimization**  
   - A* pathfinding over an OSMnx/NetworkX graph  
   - Travel‑time heuristics informed by ML predictions  
   - Dynamic rerouting to avoid predicted merge‑point delays

5. **Visualization Dashboard**  
   - Streamlit app with Folium maps and plotly charts  
   - Displays current congestion levels, forecasts, and optimized routes  
   - Filters by time of day and highlights predicted bottlenecks

---

## Key Technologies

- **TensorFlow/Keras & TensorFlow Lite** for model training and edge deployment
- **scikit-learn** for preprocessing (MinMaxScaler)
- **Django & Django REST Framework** as the backend service
- **Celery & RabbitMQ** for scheduled and asynchronous tasks
- **OSMnx & NetworkX** for road‑network graphs and bottleneck analysis
- **Streamlit & Folium** for interactive maps and dashboards
- **pandas & NumPy** for data manipulation
- **joblib** for scaler serialization

---

## Data Flow

1. **Data Collection** → TomTom API  
2. **Data Processing** → RealTimeDataProcessor  
3. **Feature Engineering** → StreamFeatureGenerator  
4. **Inference** → TFLite Model  
5. **Route Optimization** → RouteOptimizer  
6. **Visualization** → Streamlit Dashboard




---

## Full Application Showcase

A Django and React Native based full application will be presented, featuring the ML-driven traffic prediction and route optimization functionalities, real-time pothole detection, speed recommendations, and the interactive Streamlit visualizations.

## Contributing

This project is actively under development currently.

---

## License

Distributed under the MIT License. See `LICENSE` for more details.

