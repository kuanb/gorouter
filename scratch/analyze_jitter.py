#!/usr/bin/env python3
"""
Compare jittered and non-jittered routes to visualize the effect of jitter.
"""

import json
import argparse
from math import radians, sin, cos, sqrt, atan2


def haversine_distance(coord1, coord2):
    """Calculate distance between two coordinates in meters."""
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    
    R = 6371000  # Earth radius in meters
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def analyze_jitter(geojsonl_file):
    """Analyze the jitter characteristics of routes."""
    routes = []
    with open(geojsonl_file, 'r') as f:
        for line in f:
            feature = json.loads(line)
            routes.append(feature)
    
    if not routes:
        print("No routes found!")
        return
    
    jitter_meters = routes[0]['properties'].get('jitter_meters', 0)
    
    print(f"\n{'='*60}")
    print(f"Jitter Analysis: {geojsonl_file}")
    print(f"{'='*60}")
    print(f"Number of routes: {len(routes)}")
    print(f"Configured jitter: {jitter_meters} meters")
    print(f"\nPer-route segment distance statistics:")
    print(f"{'Route':<10} {'Min (m)':<12} {'Max (m)':<12} {'Avg (m)':<12} {'Std (m)':<12}")
    print("-" * 60)
    
    for route in routes:
        route_id = route['properties']['route_id']
        coords = route['geometry']['coordinates']
        
        # Calculate distances between consecutive points
        distances = []
        for i in range(len(coords) - 1):
            dist = haversine_distance(coords[i], coords[i + 1])
            distances.append(dist)
        
        if distances:
            min_dist = min(distances)
            max_dist = max(distances)
            avg_dist = sum(distances) / len(distances)
            
            # Calculate standard deviation
            variance = sum((d - avg_dist) ** 2 for d in distances) / len(distances)
            std_dist = sqrt(variance)
            
            print(f"{route_id:<10} {min_dist:<12.2f} {max_dist:<12.2f} {avg_dist:<12.2f} {std_dist:<12.2f}")
    
    print("-" * 60)
    
    # Show sample coordinates from first route
    if routes:
        print(f"\nFirst 5 coordinates of route 0:")
        coords = routes[0]['geometry']['coordinates'][:5]
        for i, (lon, lat) in enumerate(coords):
            print(f"  Point {i}: ({lon:.8f}, {lat:.8f})")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze jitter in synthetic routes'
    )
    parser.add_argument('geojsonl_file', help='Input GeoJSONl file')
    
    args = parser.parse_args()
    analyze_jitter(args.geojsonl_file)


if __name__ == '__main__':
    main()
