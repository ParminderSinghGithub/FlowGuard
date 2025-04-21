import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import requests
import os
import sys

# Add parent directory to path to import RouteOptimizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_pipeline.route_optimizer import RouteOptimizer

# --- Page Config & Welcome Message ---
st.set_page_config(
    page_title="FlowGuard - Model Visualization",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚦 FlowGuard - Model Visualization")
with st.expander("About This Dashboard", expanded=False):
    st.markdown("""
        **FlowGuard Traffic Analysis Dashboard (Simplified)**
        
        This dashboard provides real-time traffic analysis with the following features:
        
        - **Live Traffic Map**: Color-coded visualization of current traffic conditions
        - **Route Optimization**: ML-powered routing with multiple options (fastest, shortest, balanced)
        - **Traffic Patterns**: Analysis of traffic patterns based on time of day
    """)

# --- Initialize RouteOptimizer ---
@st.cache_resource
def load_route_optimizer(model_path=None):
    """Initialize RouteOptimizer with optional TFLite model"""
    try:
        return RouteOptimizer(tflite_model=model_path)
    except Exception as e:
        st.error(f"Error initializing RouteOptimizer: {str(e)}")
        return None

# Try to load the optimizer with model if available
route_optimizer = load_route_optimizer()

# --- Safe Map Rendering ---
def safe_render_map(map_obj, width=1100, height=500):
    """Safely render a folium map with error handling"""
    try:
        folium_static(map_obj, width=width, height=height)
    except Exception as e:
        st.error(f"Error rendering map: {str(e)}")
        st.info("Please try adjusting filters or refreshing the page.")

# --- Map area names to coordinates and back ---
def get_area_mapping():
    """
    Create bidirectional mapping between area names, segments and coordinates.
    This is a mock mapping for demonstration purposes.
    """
    # Map segment IDs to area names
    segment_to_area = {
        'segment_1': 'Old Ludhiana',
        'segment_2': 'Civil Lines',
        'segment_3': 'Sarabha Nagar',
        'segment_4': 'Model Town',
        'segment_5': 'Jawahar Nagar',
        'segment_6': 'Urban Estate Dugri'
    }
    
    # Map area names to coordinates (lat, lng)
    area_to_coords = {
        'Old Ludhiana': (30.911972, 75.853222),
        'Civil Lines': (30.915000, 75.855000),
        'Sarabha Nagar': (30.910000, 75.850000),
        'Model Town': (30.920000, 75.860000),
        'Jawahar Nagar': (30.905000, 75.845000),
        'Urban Estate Dugri': (30.900000, 75.840000)
    }
    
    # Map segment IDs to coordinates
    segment_to_coords = {
        'segment_1': (30.911972, 75.853222),
        'segment_2': (30.915000, 75.855000),
        'segment_3': (30.910000, 75.850000),
        'segment_4': (30.920000, 75.860000),
        'segment_5': (30.905000, 75.845000),
        'segment_6': (30.900000, 75.840000)
    }
    
    # Map coordinates to area names (for reverse lookup)
    coords_to_area = {coords: area for area, coords in area_to_coords.items()}
    
    return {
        'segment_to_area': segment_to_area,
        'area_to_coords': area_to_coords,
        'segment_to_coords': segment_to_coords, 
        'coords_to_area': coords_to_area
    }

# Get area mappings
area_mappings = get_area_mapping()

# --- Display Route with Area Information ---
def display_route_map(start_point, end_point, waypoints=None, route_info=None, route_type="default"):
    """
    Display a folium map with routing visualization using area names.
    
    Args:
        start_point: (lat, lng) tuple for starting point
        end_point: (lat, lng) tuple for ending point
        waypoints: List of waypoints
        route_info: Optional dict with route information
        route_type: String identifier for route type (for color coding)
    """
    # Select color based on route type
    route_colors = {
        "fastest": "blue",
        "shortest": "green", 
        "balanced": "purple",
        "default": "red"
    }
    color = route_colors.get(route_type, "red")
    
    # Calculate map center
    if waypoints and len(waypoints) > 0:
        all_points = [start_point] + waypoints + [end_point]
    else:
        all_points = [start_point, end_point]
        
    lat_center = np.mean([p[0] for p in all_points])
    lng_center = np.mean([p[1] for p in all_points])
    route_map = folium.Map(location=[lat_center, lng_center], zoom_start=12)
    
    # Get area names for start and end points
    start_area = area_mappings['coords_to_area'].get(start_point, "Starting Point")
    end_area = area_mappings['coords_to_area'].get(end_point, "Destination")
    
    # Add start and end markers with area names
    folium.Marker(
        location=start_point,
        icon=folium.Icon(color='green', icon='play', prefix='fa'),
        tooltip=f"Start: {start_area}"
    ).add_to(route_map)
    
    folium.Marker(
        location=end_point,
        icon=folium.Icon(color='red', icon='stop', prefix='fa'),
        tooltip=f"End: {end_area}"
    ).add_to(route_map)
    
    # Add route line
    if waypoints and len(waypoints) > 0:
        # Create waypoint path
        route_path = [start_point] + waypoints + [end_point]
        
        # Add route line
        folium.PolyLine(
            locations=route_path,
            color=color,
            weight=5,
            opacity=0.8
        ).add_to(route_map)
        
        # Add waypoint markers
        for i, waypoint in enumerate(waypoints):
            area_name = area_mappings['coords_to_area'].get(waypoint, f"Waypoint {i+1}")
            
            # Get area-specific speed if available in route_info
            speed_info = ""
            status_color = "blue"
            if route_info and 'segment_speeds' in route_info:
                segment_speed = route_info['segment_speeds'].get(i, None)
                if segment_speed is not None:
                    speed_info = f" - {segment_speed:.1f} km/h"
                    # Determine color based on speed
                    if segment_speed < 20:
                        status_color = "red"  # Heavy traffic
                    elif segment_speed < 40:
                        status_color = "orange"  # Moderate traffic
                    else:
                        status_color = "green"  # Flowing traffic
            
            # Create marker with area name and speed
            folium.CircleMarker(
                location=waypoint,
                radius=6,
                color=status_color,
                fill=True,
                fill_opacity=0.7,
                tooltip=f"{area_name}{speed_info}"
            ).add_to(route_map)
    else:
        # Direct line if no waypoints
        folium.PolyLine(
            locations=[start_point, end_point],
            color=color,
            weight=5,
            opacity=0.8
        ).add_to(route_map)
    
    safe_render_map(route_map)

# --- Load Data ---
@st.cache_data(ttl=300)
def load_data(filename: str = "traffic_data.csv") -> pd.DataFrame:
    try:
        # Get absolute path relative to current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day_name()
        df['date'] = df['timestamp'].dt.date
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(columns=['timestamp', 'hour', 'day', 'date', 'current_speed', 'latitude', 'longitude', 'road_segment_id'])

# Load data
df = load_data()

# --- Sidebar: Filters & Navigation ---
st.sidebar.header("🔍 Filters & Navigation")

# Date filter
available_dates = sorted(df['date'].unique()) if not df.empty else []
if available_dates:
    selected_date = st.sidebar.date_input(
        "Select Date", 
        value=available_dates[-1],
        min_value=available_dates[0],
        max_value=available_dates[-1]
    )
    date_filtered_df = df[df['date'] == selected_date]
else:
    date_filtered_df = df
    selected_date = datetime.now().date()
    st.sidebar.warning("No date data available")

# Hour range filter
min_hour = int(df['hour'].min()) if not df.empty else 0
max_hour = int(df['hour'].max()) if not df.empty else 23
hour_range = st.sidebar.slider(
    "Hour of Day", 
    min_hour, 
    max_hour, 
    (6, 19)  # Default business hours
)

filtered_df = date_filtered_df[
    (date_filtered_df['hour'] >= hour_range[0]) & 
    (date_filtered_df['hour'] <= hour_range[1])
] if not date_filtered_df.empty else pd.DataFrame()

# Road type filter
if not filtered_df.empty and 'road_type' in filtered_df.columns:
    road_types = ['All'] + sorted(filtered_df['road_type'].unique().tolist())
    selected_road_type = st.sidebar.selectbox("Road Type", road_types)
    if selected_road_type != 'All':
        filtered_df = filtered_df[filtered_df['road_type'] == selected_road_type]

# Navigation tabs in sidebar
page = st.sidebar.radio(
    "Navigation",
    ["Traffic Overview", "Route Optimization", "Traffic Patterns"]
)

# Display data metrics
st.sidebar.markdown("### Data Metrics")
col1, col2 = st.sidebar.columns(2)
col1.metric("Data Points", f"{len(filtered_df):,}" if not filtered_df.empty else "0")
if not filtered_df.empty:
    col2.metric("Avg Speed", f"{filtered_df['current_speed'].mean():.1f} km/h" if 'current_speed' in filtered_df.columns else "N/A")

# --- TRAFFIC OVERVIEW PAGE ---
if page == "Traffic Overview":
    st.header("🌐 Live Traffic Map")
    
    with st.expander("Display Map", expanded=True):
        if not filtered_df.empty and 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
            # Get map center
            center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
            m = folium.Map(location=center, zoom_start=12)
            
            # Create marker cluster for traffic status
            marker_cluster = MarkerCluster().add_to(m)
            
            for _, row in filtered_df.tail(300).iterrows():
                # Determine color based on speed
                if row['current_speed'] < 20:
                    color = 'red'
                    status = 'Heavy Traffic'
                elif row['current_speed'] < 40:
                    color = 'orange'
                    status = 'Moderate Traffic'
                else:
                    color = 'green'
                    status = 'Flowing'
                
                # Create simple popup content
                popup_html = f"""
                <div style="width: 200px">
                    <h4>Traffic Data</h4>
                    <b>Speed:</b> {float(row['current_speed']):.1f} km/h<br>
                    <b>Status:</b> {status}<br>
                </div>
                """
                
                folium.CircleMarker(
                    location=(row['latitude'], row['longitude']),
                    radius=5,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=250)
                ).add_to(marker_cluster)
            
            safe_render_map(m)
        else:
            st.warning("No location data available for the selected filters")
    
    # Speed trends
    st.header("📊 Speed Trends")
    
    if not filtered_df.empty and 'current_speed' in filtered_df.columns:
        # Create hourly aggregation
        hourly_df = filtered_df.copy()
        hourly_df['hour_bin'] = hourly_df['timestamp'].dt.floor('H')
        hourly_speed = hourly_df.groupby('hour_bin')['current_speed'].mean().reset_index()
        
        # Comparison with previous day (if available)
        prev_day = selected_date - timedelta(days=1)
        prev_day_data = df[df['date'] == prev_day] if not df.empty else pd.DataFrame()
        
        if not prev_day_data.empty:
            prev_hourly = prev_day_data.copy()
            prev_hourly['hour_bin'] = prev_hourly['timestamp'].dt.floor('H')
            prev_hourly = prev_hourly.groupby('hour_bin')['current_speed'].mean().reset_index()
            
            # Align timestamps to compare by hour
            prev_hourly['hour'] = prev_hourly['hour_bin'].dt.hour
            hourly_speed['hour'] = hourly_speed['hour_bin'].dt.hour
            
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(hourly_speed['hour'], hourly_speed['current_speed'], marker='o', linewidth=2, label='Today')
            ax.plot(prev_hourly['hour'], prev_hourly['current_speed'], marker='x', linestyle='--', linewidth=2, label='Yesterday')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Average Speed (km/h)')
            ax.set_title('Speed Trend Comparison')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            st.pyplot(fig)
        else:
            # Show only today's data
            st.line_chart(hourly_speed.set_index('hour_bin')['current_speed'])
    else:
        st.warning("No speed data available for the selected date/time range")

    # Daily statistics
    if not filtered_df.empty and 'current_speed' in filtered_df.columns:
        st.subheader("Daily Statistics")
        # Get busiest hour
        hourly_count = filtered_df.groupby(filtered_df['timestamp'].dt.hour).size()
        busiest_hour = hourly_count.idxmax() if not hourly_count.empty else "N/A"
        
        metrics_col1, metrics_col2 = st.columns(2)
        metrics_col1.metric("Busiest Hour", f"{busiest_hour}:00" if busiest_hour != "N/A" else "N/A")
        metrics_col2.metric("Avg Speed", f"{filtered_df['current_speed'].mean():.1f} km/h")

# --- ROUTE OPTIMIZATION PAGE ---
elif page == "Route Optimization":
    st.header("🚗 Route Optimization")
    
    st.markdown(
        """
        Select your start and end locations to find the optimal route with real-time traffic predictions.
        """
    )
    
    # Get area names from mapping
    served_areas = area_mappings['area_to_coords']
    
    # Route configuration
    with st.form("route_planning_form"):
        st.subheader("Route Planning")
        
        route_col1, route_col2, route_col3 = st.columns([1, 1, 1])
        
        with route_col1:
            start_area = st.selectbox("Start Area", list(served_areas.keys()))
        
        with route_col2:
            end_area = st.selectbox("End Area", list(served_areas.keys()), index=1 if len(served_areas) > 1 else 0)
        
        with route_col3:
            depart_time = st.time_input("Departure Time", value=datetime.now().time())
        
        # Add travel preferences
        priority = st.radio(
            "Route Priority",
            ["Fastest Route", "Balanced Route", "Shortest Route"]
        )
        
        calculate = st.form_submit_button("Find Route")
    
    # Process routing request
    if calculate:
        # Get coordinates from selected areas
        start_coords = served_areas[start_area]
        end_coords = served_areas[end_area]
        
        # Combine date and time for departure
        departure_datetime = datetime.combine(
            datetime.now().date(), 
            depart_time
        )
        
        # Create route tabs based on priority
        route_tabs = st.tabs([priority])
        
        with route_tabs[0]:
            # For demonstration, we'll create mock route data with speed predictions
            # In a real app, this would use the RouteOptimizer
            
            # Generate sample segment speeds (mock data)
            segment_speeds = {
                0: 45.3,  # Speed for first segment
                1: 28.7,  # Speed for second segment
                2: 52.1,  # Speed for third segment
                3: 35.9   # Speed for fourth segment
            }
            
            # Generate waypoints based on a simple path between areas
            # In real app these would come from the route optimizer
            waypoints = []
            
            # Check if we're not going directly from neighbors
            # For simplicity, add intermediate points
            waypoints = []

            # Get all areas excluding start and end points
            available_areas = [area for area in served_areas.keys() 
                              if area != start_area and area != end_area]
            
            # Create a mock "distance" matrix between areas (for demonstration)
            # In a real implementation, this would use actual distances or come from RouteOptimizer
            def mock_distance_between(area1, area2):
                """Calculate mock distance between two areas based on coordinates"""
                coord1 = served_areas[area1]
                coord2 = served_areas[area2]
                # Simple approximation of distance
                return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5
            
            # Determine intermediate areas that make sense for the route
            # Sort areas by their position along the path from start to end
            route_relevant_areas = sorted(
                available_areas,
                key=lambda area: (
                    mock_distance_between(start_area, area) + 
                    mock_distance_between(area, end_area)
                )
            )
            
            # Select waypoints based on route priority but use the same pool of areas
            if priority == "Fastest Route":
                # Fastest route - use only the most direct intermediate point if available
                if route_relevant_areas:
                    waypoints = [served_areas[route_relevant_areas[0]]]
            elif priority == "Balanced Route":
                # Balanced route - use 1-2 good intermediate points
                if len(route_relevant_areas) >= 2:
                    waypoints = [
                        served_areas[route_relevant_areas[0]],
                        served_areas[route_relevant_areas[1]]
                    ]
                elif route_relevant_areas:
                    waypoints = [served_areas[route_relevant_areas[0]]]
            elif priority == "Shortest Route":
                # Shortest route might not be the most direct
                # So pick points that create a path with more detail
                if len(route_relevant_areas) >= 3:
                    waypoints = [
                        served_areas[route_relevant_areas[0]],
                        served_areas[route_relevant_areas[1]],
                        served_areas[route_relevant_areas[2]]
                    ]
                elif len(route_relevant_areas) >= 2:
                    waypoints = [
                        served_areas[route_relevant_areas[0]],
                        served_areas[route_relevant_areas[1]]
                    ]
                elif route_relevant_areas:
                    waypoints = [served_areas[route_relevant_areas[0]]]
                else:
                    waypoints = []
                    
            # Create route info with speeds
            route_info = {
                'segment_speeds': segment_speeds,
                'total_time': "00:25:18",
                'total_distance': 8.7
            }
            
            # Display route metrics
            col1, col2 = st.columns(2)
            col1.metric("Estimated Time", route_info['total_time'])
            col2.metric("Distance", f"{route_info['total_distance']:.1f} km")
            
            # Display the map with waypoints and speed indicators
            display_route_map(
                start_coords, 
                end_coords, 
                waypoints, 
                route_info, 
                route_type=priority.split()[0].lower()
            )
            
            # Display speed suggestions in a clean format
            st.subheader("🚦 Speed Suggestions")
            
            # Create a clean visualization of speed suggestions
            speed_data = []
            for i, waypoint in enumerate(waypoints):
                area_name = area_mappings['coords_to_area'].get(waypoint, f"Segment {i+1}")
                speed = segment_speeds.get(i, 0)
                
                # Determine traffic status
                if speed < 20:
                    status = "Heavy Traffic"
                    emoji = "🔴"
                elif speed < 40:
                    status = "Moderate Traffic"
                    emoji = "🟠"
                else:
                    status = "Flowing Traffic"
                    emoji = "🟢"
                
                speed_data.append({
                    "Area": area_name,
                    "Speed": speed,
                    "Status": status,
                    "Emoji": emoji
                })
            
            # Display speed data in a clean table
            if speed_data:
                speed_df = pd.DataFrame(speed_data)
                
                # Create custom formatting
                for i, row in enumerate(speed_data):
                    cols = st.columns([1, 1, 1])
                    cols[0].markdown(f"**{row['Area']}**")
                    cols[1].markdown(f"**{row['Speed']:.1f} km/h**")
                    cols[2].markdown(f"**{row['Emoji']} {row['Status']}**")
                    
                    # Add separator
                    if i < len(speed_data) - 1:
                        st.markdown("---")
            else:
                st.info("No detailed speed information available for this route")

# --- TRAFFIC PATTERNS PAGE ---
elif page == "Traffic Patterns":
    st.header("📊 Traffic Pattern Analysis")
    
    if not filtered_df.empty and 'current_speed' in filtered_df.columns:
        # Create hourly pattern
        if 'hour' not in filtered_df.columns:
            filtered_df['hour'] = filtered_df['timestamp'].dt.hour
            
        hourly_pattern = filtered_df.groupby('hour')['current_speed'].mean().reset_index()
        
        st.subheader("Hourly Speed Pattern")
        st.bar_chart(hourly_pattern.set_index('hour'))
        
        # Show time range comparisons
        st.subheader("Time Range Comparison")
        
        # Define time ranges
        time_ranges = {
            "Morning Rush (7-10)": (7, 10),
            "Midday (11-14)": (11, 14),
            "Evening Rush (16-19)": (16, 19),
            "Night (20-23)": (20, 23)
        }
        
        # Calculate average speeds for each time range
        time_range_speeds = []
        for range_name, (start, end) in time_ranges.items():
            range_df = filtered_df[(filtered_df['hour'] >= start) & (filtered_df['hour'] <= end)]
            if len(range_df) > 0:
                avg_speed = range_df['current_speed'].mean()
                time_range_speeds.append({
                    'Time Range': range_name,
                    'Average Speed': avg_speed
                })
        
        # Display as bar chart
        if time_range_speeds:
            time_range_df = pd.DataFrame(time_range_speeds)
            st.bar_chart(time_range_df.set_index('Time Range'))
        else:
            st.info("No data available for time range comparison")
        
        # Weekly pattern if day column exists
        if 'day' in filtered_df.columns:
            st.subheader("Day of Week Pattern")
            
            # Order days of week properly
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_pattern = filtered_df.groupby('day')['current_speed'].mean().reset_index()
            
            # Ensure all days are in the expected order
            day_dict = {day: daily_pattern[daily_pattern['day'] == day]['current_speed'].mean() 
                     if day in daily_pattern['day'].values else 0 
                     for day in day_order}
            
            day_df = pd.DataFrame({
                'day': list(day_dict.keys()),
                'current_speed': list(day_dict.values())
            })
            
            st.bar_chart(day_df.set_index('day'))
        
        # Traffic trends over time
        st.subheader("Traffic Trends Over Time")
        date_pattern = filtered_df.copy()
        date_pattern['date_hour'] = date_pattern['timestamp'].dt.floor('H')
        date_speed = date_pattern.groupby('date_hour')['current_speed'].mean().reset_index()
        
        st.line_chart(date_speed.set_index('date_hour'))
        
        # Traffic distribution
        st.subheader("Traffic Speed Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(filtered_df['current_speed'], bins=20, alpha=0.7, color='blue')
        ax.set_xlabel('Speed (km/h)')
        ax.set_ylabel('Frequency')
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
    else:
        st.warning("No data available for pattern analysis")