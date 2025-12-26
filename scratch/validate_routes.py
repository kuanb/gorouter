#!/usr/bin/env python3
"""
Validate and show statistics for generated synthetic routes.
"""

import json
import argparse
from math import radians, sin, cos, sqrt, atan2


def haversine_distance(coord1, coord2):
    """Calculate distance between two coordinates in miles."""
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    
    R = 3958.8  # Earth radius in miles
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def calculate_route_distance(coordinates):
    """Calculate total distance of a route."""
    total = 0.0
    for i in range(len(coordinates) - 1):
        total += haversine_distance(coordinates[i], coordinates[i + 1])
    return total


def main():
    parser = argparse.ArgumentParser(description='Validate synthetic routes')
    parser.add_argument('geojsonl_file', help='Input GeoJSONl file')
    args = parser.parse_args()
    
    routes = []
    with open(args.geojsonl_file, 'r') as f:
        for line in f:
            feature = json.loads(line)
            routes.append(feature)
    
    print(f"Loaded {len(routes)} routes\n")
    print(f"{'Route':<10} {'Probes':<10} {'Distance (mi)':<15} {'Avg Speed (mph)':<15}")
    print("-" * 60)
    
    distances = []
    for route in routes:
        route_id = route['properties']['route_id']
        num_probes = route['properties']['num_probes']
        coords = route['geometry']['coordinates']
        
        distance = calculate_route_distance(coords)
        distances.append(distance)
        
        # Assuming 30mph average speed
        avg_speed = 30.0  # mph
        
        print(f"{route_id:<10} {num_probes:<10} {distance:<15.3f} {avg_speed:<15.1f}")
    
    print("-" * 60)
    print(f"\nStatistics:")
    print(f"  Min distance: {min(distances):.3f} miles")
    print(f"  Max distance: {max(distances):.3f} miles")
    print(f"  Avg distance: {sum(distances)/len(distances):.3f} miles")


if __name__ == '__main__':
    main()
