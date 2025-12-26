#!/usr/bin/env python3
"""
Generate synthetic routes from OSM PBF data.

Creates linestrings with exactly 180 coordinate probes that follow roads
(tertiary, secondary, primary, motorway, trunk and their links) over 1-2 mile distances.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

try:
    import osmium
    from shapely.geometry import LineString, Point
    from shapely.ops import linemerge
    import pyproj
except ImportError as e:
    print(f"Missing required package: {e}")
    print("\nInstall dependencies with:")
    print("pip install osmium shapely pyproj")
    exit(1)


class RoadNetworkHandler(osmium.SimpleHandler):
    """Extract road network from OSM PBF."""
    
    ROAD_TYPES = {
        'motorway', 'motorway_link',
        'trunk', 'trunk_link',
        'primary', 'primary_link',
        'secondary', 'secondary_link',
        'tertiary', 'tertiary_link'
    }
    
    def __init__(self):
        super().__init__()
        self.ways = []
        self.nodes = {}
        
    def node(self, n):
        """Store node locations."""
        self.nodes[n.id] = (n.location.lon, n.location.lat)
    
    def way(self, w):
        """Extract relevant road ways."""
        tags = {tag.k: tag.v for tag in w.tags}
        highway = tags.get('highway', '')
        
        if highway in self.ROAD_TYPES:
            # Store way with its nodes
            nodes = [n.ref for n in w.nodes]
            self.ways.append({
                'id': w.id,
                'highway': highway,
                'nodes': nodes,
                'tags': tags
            })


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculate distance between two coordinates in miles."""
    from math import radians, sin, cos, sqrt, atan2
    
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    
    R = 3958.8  # Earth radius in miles
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def add_jitter_to_point(lon: float, lat: float, jitter_meters: float) -> Tuple[float, float]:
    """Add random jitter to a coordinate point.
    
    Args:
        lon: Longitude
        lat: Latitude  
        jitter_meters: Maximum jitter distance in meters
    
    Returns:
        Jittered (lon, lat) tuple
    """
    if jitter_meters <= 0:
        return (lon, lat)
    
    # Random angle and distance
    angle = random.uniform(0, 2 * 3.14159265359)
    distance = random.uniform(0, jitter_meters)
    
    # Convert meters to degrees (approximate)
    # At the equator: 1 degree ≈ 111,320 meters
    # Adjust longitude by latitude to account for convergence
    import math
    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * math.cos(math.radians(lat))
    
    delta_lat = (distance * math.cos(angle)) / meters_per_degree_lat
    delta_lon = (distance * math.sin(angle)) / meters_per_degree_lon
    
    return (lon + delta_lon, lat + delta_lat)


def interpolate_linestring(coords: List[Tuple[float, float]], 
                          num_points: int = 180, 
                          jitter_meters: float = 0.0) -> List[Tuple[float, float]]:
    """Interpolate a linestring to exactly num_points coordinates with optional jitter.
    
    Args:
        coords: List of (lon, lat) coordinate tuples
        num_points: Number of points to generate
        jitter_meters: Random jitter to add to each point in meters (default: 0)
    
    Returns:
        List of interpolated and jittered coordinates
    """
    if len(coords) < 2:
        return coords
    
    line = LineString(coords)
    total_length = line.length
    
    # Generate points at equal intervals along the line
    distances = [i * total_length / (num_points - 1) for i in range(num_points)]
    interpolated = [line.interpolate(d).coords[0] for d in distances]
    
    # Add jitter if requested
    if jitter_meters > 0:
        interpolated = [add_jitter_to_point(lon, lat, jitter_meters) 
                       for lon, lat in interpolated]
    
    return interpolated


def build_road_graph(ways: List[Dict], nodes: Dict) -> Dict:
    """Build a graph of connected road segments."""
    graph = {}  # node_id -> list of (neighbor_id, way_id, coords)
    
    for way in ways:
        way_nodes = way['nodes']
        coords = []
        
        # Get coordinates for this way
        for node_id in way_nodes:
            if node_id in nodes:
                coords.append(nodes[node_id])
        
        if len(coords) < 2:
            continue
        
        # Add edges in both directions
        for i in range(len(way_nodes) - 1):
            node_a = way_nodes[i]
            node_b = way_nodes[i + 1]
            
            if node_a in nodes and node_b in nodes:
                coord_a = nodes[node_a]
                coord_b = nodes[node_b]
                
                if node_a not in graph:
                    graph[node_a] = []
                if node_b not in graph:
                    graph[node_b] = []
                
                graph[node_a].append((node_b, way['id'], [coord_a, coord_b]))
                graph[node_b].append((node_a, way['id'], [coord_b, coord_a]))
    
    return graph


def generate_random_route(graph: Dict, nodes: Dict, 
                          min_distance: float = 1.0, 
                          max_distance: float = 2.0,
                          max_attempts: int = 200) -> List[Tuple[float, float]]:
    """Generate a random route along the road network."""
    if not graph:
        return []
    
    for attempt in range(max_attempts):
        # Pick a random starting node with good connectivity
        start_candidates = [n for n in graph.keys() if len(graph[n]) >= 2]
        if not start_candidates:
            start_candidates = list(graph.keys())
        
        start_node = random.choice(start_candidates)
        
        route_coords = [nodes[start_node]]
        visited_edges = set()
        current_node = start_node
        total_distance = 0.0
        
        # Walk the graph until we reach target distance
        target = random.uniform(min_distance, max_distance)
        
        while total_distance < target * 1.5:  # Allow overshoot for later trimming
            if current_node not in graph or not graph[current_node]:
                break
            
            # Get all available neighbors
            available = []
            for next_node, way_id, edge_coords in graph[current_node]:
                edge_key = (min(current_node, next_node), max(current_node, next_node), way_id)
                if edge_key not in visited_edges:
                    available.append((next_node, way_id, edge_coords, edge_key))
            
            # If no unvisited neighbors, allow revisiting
            if not available:
                for next_node, way_id, edge_coords in graph[current_node]:
                    edge_key = (min(current_node, next_node), max(current_node, next_node), way_id)
                    available.append((next_node, way_id, edge_coords, edge_key))
            
            if not available:
                break
            
            # Pick a random neighbor
            next_node, way_id, edge_coords, edge_key = random.choice(available)
            visited_edges.add(edge_key)
            
            # Calculate this segment's distance
            segment_dist = sum(
                haversine_distance(edge_coords[i], edge_coords[i+1]) 
                for i in range(len(edge_coords)-1)
            )
            
            # If adding this would go too far over, consider stopping
            if total_distance >= min_distance and total_distance + segment_dist > max_distance * 1.3:
                if total_distance >= min_distance * 0.9:
                    return route_coords
            
            # Add the edge coordinates (skip first to avoid duplication)
            for coord in edge_coords[1:]:
                route_coords.append(coord)
            
            total_distance += segment_dist
            current_node = next_node
            
            # Check if we're in acceptable range
            if min_distance * 0.9 <= total_distance <= max_distance * 1.1:
                return route_coords
            
            # Prevent infinite routes
            if len(route_coords) > 3000:
                break
        
        # Accept route if within relaxed tolerance
        if min_distance * 0.7 <= total_distance <= max_distance * 1.3:
            return route_coords
    
    return []


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic routes from OSM PBF data'
    )
    parser.add_argument('pbf_file', help='Input OSM PBF file')
    parser.add_argument('-n', '--num-routes', type=int, default=10,
                       help='Number of routes to generate (default: 10)')
    parser.add_argument('-o', '--output', default='synthetic_routes.geojsonl',
                       help='Output newline-delimited GeoJSON file (default: synthetic_routes.geojsonl)')
    parser.add_argument('--min-distance', type=float, default=1.0,
                       help='Minimum route distance in miles (default: 1.0)')
    parser.add_argument('--max-distance', type=float, default=2.0,
                       help='Maximum route distance in miles (default: 2.0)')
    parser.add_argument('--num-probes', type=int, default=180,
                       help='Number of coordinate probes per route (default: 180)')
    parser.add_argument('--jitter-meters', type=float, default=10.0,
                       help='Random jitter to add to each point in meters (default: 10.0, use 0 to disable)')
    
    args = parser.parse_args()
    
    print(f"Reading OSM PBF file: {args.pbf_file}")
    
    # Extract road network
    handler = RoadNetworkHandler()
    handler.apply_file(args.pbf_file)
    
    print(f"Extracted {len(handler.ways)} road segments")
    print(f"Found {len(handler.nodes)} nodes")
    
    # Build graph
    print("Building road network graph...")
    graph = build_road_graph(handler.ways, handler.nodes)
    print(f"Graph has {len(graph)} connected nodes")
    
    if not graph:
        print("Error: No connected road network found!")
        return
    
    # Generate routes
    print(f"\nGenerating {args.num_routes} synthetic routes...")
    output_path = Path(args.output)
    
    successful = 0
    with open(output_path, 'w') as f:
        for i in range(args.num_routes):
            log_every = max(1, args.num_routes // 50)
            if (i + 1) % log_every == 0 or i == 0 or (i + 1) == args.num_routes:
                print(f"  Generated {i + 1}/{args.num_routes} routes...")
            
            route_coords = generate_random_route(
                graph, handler.nodes,
                min_distance=args.min_distance,
                max_distance=args.max_distance
            )
            
            if not route_coords:
                continue
            
            # Interpolate to exact number of probes with jitter
            interpolated = interpolate_linestring(route_coords, args.num_probes, args.jitter_meters)
            
            # Create GeoJSON feature
            feature = {
                "type": "Feature",
                "properties": {
                    "route_id": i,
                    "num_probes": len(interpolated),
                    "original_points": len(route_coords),
                    "jitter_meters": args.jitter_meters
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": interpolated
                }
            }
            
            f.write(json.dumps(feature) + '\n')
            successful += 1
    
    print(f"\n✓ Successfully generated {successful} routes")
    print(f"✓ Saved to: {output_path}")
    
    if successful < args.num_routes:
        print(f"⚠ Warning: Only generated {successful}/{args.num_routes} routes")
        print("  (Some routes may not have met distance requirements)")


if __name__ == '__main__':
    main()
