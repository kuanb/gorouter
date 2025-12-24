# gosm-router

A simple routing and map matching library for auto traces built in Go.

## Overview

This project provides routing and GPS trace matching algorithms designed specifically for car navigation. It operates on OpenStreetMap data extracted as PBF files (metro/region extracts).

## Features

- **OSM PBF Parsing**: Load and process OpenStreetMap PBF extracts
- **Road Network Graph**: Builds a simplified graph from highway segments
- **HMM Map Matching**: Match GPS coordinates to road segments using a Hidden Markov Model

## Usage

```go
// Load OSM data
graph := osm.LoadOsmFile("./data/district-of-columbia.osm.pbf")

// Create map matcher
matcher := routing.NewHMMMapMatcher(graph)

// Match GPS coordinates to roads
coords := []routing.Coordinate{
    {Lon: -77.0239, Lat: 38.9152},
    {Lon: -77.0239, Lat: 38.9154},
}
result := matcher.Match(coords)
```

## Data

Place your OSM PBF extract in the `data/` directory. You can obtain metro extracts from sources like [Geofabrik](https://download.geofabrik.de/).

## Supported Road Types

The router filters to car-navigable highways:
- motorway, trunk, primary, secondary, tertiary
- residential, service, living_street
- Associated link roads (*_link)

