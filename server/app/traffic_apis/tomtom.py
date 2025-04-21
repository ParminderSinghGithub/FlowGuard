# app/traffic_apis/tomtom.py
import requests
import logging
import time
from django.conf import settings
from django.utils import timezone
from math import radians, sin, cos, sqrt, atan2
import osmnx as ox
from typing import List, Dict, Any
import networkx as nx

logger = logging.getLogger(__name__)


TOMTOM_BASE_URL = "https://api.tomtom.com/traffic/services/4"
API_KEY = settings.TOMTOM_API_KEY

# Ludhiana hotspots (lat, lon, radius_km)
LUDHIANA_HOTSPOTS = [
    (30.9000, 75.8573, 3),  # City Center (Ferozepur Rd)
    (30.9158, 75.8227, 2),  # PAU/Sarabha Nagar
    (30.8412, 75.8573, 2),  # Bus Stand
    (30.8786, 75.8000, 1.5)  # Dugri Rd
]

def get_road_connections(lat: float, lon: float, radius_km: float) -> List[str]:
    """
    Fetch connected roads using OSMnx and convert to our road_id format.
    Returns list of connected road IDs within the radius.
    """
    try:
        # Get the road network graph
        G = ox.graph_from_point((lat, lon), dist=radius_km*1000, network_type='drive')
        
        # Get the nearest node to our hotspot
        center_node = ox.distance.nearest_nodes(G, lon, lat)
        
        # Get all connected nodes within radius
        subgraph = nx.ego_graph(G, center_node, radius=radius_km*1000)
        connected_nodes = list(subgraph.nodes())
        
        # Convert to our road_id format (LUD_lat_lon)
        connections = []
        for node in connected_nodes:
            node_data = G.nodes[node]
            connections.append(f"LUD_{node_data['y']:.4f}_{node_data['x']:.4f}")
            
        # Remove duplicates and self-connection
        connections = list(set(connections))
        connections = [c for c in connections if c != f"LUD_{lat:.4f}_{lon:.4f}"]
        
        return connections[:10]  # Limit to 10 closest connections
        
    except Exception as e:
        logger.error(f"OSMnx connection error for {lat},{lon}: {str(e)}")
        return []

def is_bottleneck(G, node: int, current_speed: float, free_flow_speed: float, road_type: str) -> bool:
    """
    Enhanced Ludhiana-specific bottleneck detection combining graph structure, speed data, and time of day.
    """

    # 1. Graph-based checks
    if G.degree(node) > 2:  # Merging or major intersection
        return True

    # 2. Speed-based checks (road type-specific thresholds)
    if free_flow_speed <= 0:
        return False
        
    speed_ratio = current_speed / free_flow_speed

    if road_type == 'FRC2':  # Major roads
        if speed_ratio < 0.65:
            return True
    elif road_type == 'FRC3':  # Local roads
        if speed_ratio < 0.55:
            return True
    else:  # FRC6 and others
        if speed_ratio < 0.5:
            return True

    # 3. Time-based checks (rush hours or nighttime)
    hour = timezone.now().hour
    if (2 <= hour <= 6 or 12 <= hour <= 20) and speed_ratio < 0.8:
        return True

    return False

# tomtom.py
def get_ludhiana_traffic() -> List[Dict[str, Any]]:
    """Robust API fetcher with connection mapping"""
    results = []
    
    for idx, (lat, lon, radius) in enumerate(LUDHIANA_HOTSPOTS):
        try:
            response = requests.get(
                f"{TOMTOM_BASE_URL}/flowSegmentData/relative0/10/json",
                params={
                    'point': f"{lat},{lon}",
                    'radius': radius * 1000,
                    'key': API_KEY,
                    'zoom': 12
                },
                timeout=15
            )
            response.raise_for_status()
            
            segment_data = response.json().get('flowSegmentData', {})
            if not segment_data:
                continue

            # Calculate congestion using free flow speed
            free_flow = max(segment_data.get('freeFlowSpeed', 1), 1)  # Prevent division by zero
            current = segment_data.get('currentSpeed', 0)
            
            results.append({
                'coordinates': segment_data.get('coordinates', {}),
                'speeds': {
                    'current': current,
                    'free_flow': free_flow
                },
                'road_type': segment_data.get('frc', 'UNKNOWN'),
                'congestion_level': 'severe' if current / free_flow < 0.4 else 'moderate',
                'is_bottleneck': current / free_flow < 0.5
            })

        except requests.exceptions.RequestException as e:
            logger.error(f"Hotspot {idx+1} failed: {str(e)}")
            continue
            
    return results