# Synthetic Route Generator for OSM Data

Generate synthetic GPS routes from OpenStreetMap PBF files that follow realistic road networks.

## Features

- Extracts road networks from OSM PBF files (motorway, trunk, primary, secondary, tertiary, and their links)
- Generates random routes that follow actual roads
- Interpolates routes to exactly 180 coordinate probes (configurable)
- Adds realistic GPS jitter/noise to coordinates (default: 10 meters, configurable)
- Targets 1-2 mile distances (representing ~30mph speeds)
- Outputs newline-delimited GeoJSON for easy processing

## Installation

```bash
pip install osmium shapely pyproj
```

Or use the provided requirements file:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Generate 10 synthetic routes (default):

```bash
python generate_synthetic_routes.py your_area.osm.pbf
```

### Custom Number of Routes

Generate 50 routes:

```bash
python generate_synthetic_routes.py your_area.osm.pbf -n 50
```

### Custom Output File

```bash
python generate_synthetic_routes.py your_area.osm.pbf -n 100 -o my_routes.geojsonl
```

### Custom Distance Range

Generate routes between 0.5 and 1.5 miles:

```bash
python generate_synthetic_routes.py your_area.osm.pbf --min-distance 0.5 --max-distance 1.5
```

### Custom Number of Probes

Generate routes with 360 probes instead of 180:

```bash
python generate_synthetic_routes.py your_area.osm.pbf --num-probes 360
```

### Custom Jitter Amount

Add 20 meters of random jitter (simulating GPS noise):

```bash
python generate_synthetic_routes.py your_area.osm.pbf --jitter-meters 20
```

Disable jitter completely:

```bash
python generate_synthetic_routes.py your_area.osm.pbf --jitter-meters 0
```

## Command-Line Options

```
usage: generate_synthetic_routes.py [-h] [-n NUM_ROUTES] [-o OUTPUT]
                                     [--min-distance MIN_DISTANCE]
                                     [--max-distance MAX_DISTANCE]
                                     [--num-probes NUM_PROBES]
                                     pbf_file

positional arguments:
  pbf_file              Input OSM PBF file

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_ROUTES, --num-routes NUM_ROUTES
                        Number of routes to generate (default: 10)
  -o OUTPUT, --output OUTPUT
                        Output newline-delimited GeoJSON file
                        (default: synthetic_routes.geojsonl)
  --min-distance MIN_DISTANCE
                        Minimum route distance in miles (default: 1.0)
  --max-distance MAX_DISTANCE
                        Maximum route distance in miles (default: 2.0)
  --num-probes NUM_PROBES
                        Number of coordinate probes per route (default: 180)
  --jitter-meters JITTER_METERS
                        Random jitter to add to each point in meters 
                        (default: 10.0, use 0 to disable)
```

## Output Format

The script generates a newline-delimited GeoJSON file where each line is a complete GeoJSON Feature:

```json
{
  "type": "Feature",
  "properties": {
    "route_id": 0,
    "num_probes": 180,
    "original_points": 96,
    "jitter_meters": 10.0
  },
  "geometry": {
    "type": "LineString",
    "coordinates": [
      [-77.0416546, 38.9129252],
      [-77.04165454375665, 38.912831892274866],
      ...
    ]
  }
}
```

## Validation

Use the included validation script to check your generated routes:

```bash
python validate_routes.py synthetic_routes.geojsonl
```

This will show:
- Number of probes per route
- Actual distance of each route
- Statistics (min, max, avg distances)

## Jitter Analysis

Use the jitter analysis script to see the effect of GPS noise:

```bash
python analyze_jitter.py synthetic_routes.geojsonl
```

This will show:
- Configured jitter amount
- Distance statistics between consecutive probes
- Sample coordinates

Example output comparing no jitter vs 10m jitter:
```
# No jitter (--jitter-meters 0)
Route      Min (m)      Max (m)      Avg (m)      Std (m)     
0          2.72         9.51         8.15         0.55

# With 10m jitter (default)
Route      Min (m)      Max (m)      Avg (m)      Std (m)     
0          0.09         23.34        10.31        4.98
```

## Example Output

```
Loaded 20 routes

Route      Probes     Distance (mi)   Avg Speed (mph)
------------------------------------------------------------
0          180        0.897           30.0           
1          180        0.897           30.0           
2          180        0.900           30.0           
...
------------------------------------------------------------

Statistics:
  Min distance: 0.811 miles
  Max distance: 0.994 miles
  Avg distance: 0.898 miles
```

## How It Works

1. **Extract Road Network**: Parses the OSM PBF file and extracts all roads of the specified types
2. **Build Graph**: Creates a connectivity graph of road segments
3. **Random Walk**: Performs random walks along the graph starting from random nodes
4. **Distance Targeting**: Continues walking until the route reaches the target distance range
5. **Interpolation**: Uses linear interpolation along the route to generate exactly N probes
6. **Jitter Application**: Adds random GPS-like noise to each coordinate (default 10m)
7. **Output**: Saves each route as a GeoJSON LineString in newline-delimited format

## Notes

- The script uses a relaxed tolerance (~70-130% of target distance) to ensure route generation succeeds
- Routes are generated by random walks, so they represent realistic driving patterns
- The interpolation ensures equal spacing of probes along the route
- Each route attempts to avoid revisiting the same road segments when possible
- Jitter is applied using a random angle and distance (0 to max_jitter_meters) from each interpolated point
- Jitter simulates realistic GPS noise and makes routes more suitable for testing navigation systems

## Tested With

- Washington DC OSM PBF (district-of-columbia-251222_osm.pbf)
- Generated 20 routes successfully with 180 probes each
- Average route distance: ~0.9 miles
