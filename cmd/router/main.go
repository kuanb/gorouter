package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"

	"kuanb/gosm-router/osm"
	"kuanb/gosm-router/routing"
)

type GeoJSONFeatureCollection struct {
	Type     string           `json:"type"`
	Features []GeoJSONFeature `json:"features"`
}

type GeoJSONFeature struct {
	Type       string          `json:"type"`
	Properties json.RawMessage `json:"properties"`
	Geometry   GeoJSONGeometry `json:"geometry"`
}

type GeoJSONGeometry struct {
	Type        string      `json:"type"`
	Coordinates [][]float64 `json:"coordinates"`
}

func main() {
	fmt.Println("gosm-router")
	graph := osm.LoadOsmFile("./data/district-of-columbia-251222.osm.pbf")
	fmt.Printf("Nodes: %d, Ways: %d\n", len(graph.Nodes), len(graph.Ways))

	geojsonStr := `{ "type": "FeatureCollection", "features": [ { "type": "Feature", "properties": {}, "geometry": { "coordinates": [ [ -77.02397423008418, 38.91520521011586 ], [ -77.02397423008418, 38.91545343777963 ], [ -77.02387755496437, 38.915596356344 ], [ -77.02320082912921, 38.915577551286276 ] ], "type": "LineString" } } ] }`

	var fc GeoJSONFeatureCollection
	if err := json.Unmarshal([]byte(geojsonStr), &fc); err != nil {
		log.Fatal(err)
	}

	var coords []routing.Coordinate
	for _, feature := range fc.Features {
		for _, coord := range feature.Geometry.Coordinates {
			coords = append(coords, routing.Coordinate{
				Lon: coord[0],
				Lat: coord[1],
			})
		}
	}
	fmt.Printf("Parsed %d coordinates\n", len(coords))

	matcher := routing.NewHMMMapMatcher(graph)
	match := matcher.Match(coords)
	fmt.Printf("match: %+v\n", match)

	// Utility function to generate a random color in hex format (#RRGGBB)
	randomColor := func() string {
		const letters = "0123456789ABCDEF"
		b := make([]byte, 7)
		b[0] = '#'
		for i := 1; i < 7; i++ {
			// Use true randomness for each digit
			b[i] = letters[rand.Intn(16)]
		}
		log.Printf("randomColor: %s", string(b))
		return string(b)
	}

	features := make([]interface{}, 0, len(match.MatchedWays))

	for _, wayID := range match.MatchedWays {
		way := graph.Ways[int64(wayID)]
		if way == nil || len(way.Geometry) < 2 {
			continue
		}

		// Build coordinates slice for the geometry (orb.LineString is [][]float64)
		lineCoords := make([][]float64, 0, len(way.Geometry))
		for _, pt := range way.Geometry {
			lineCoords = append(lineCoords, []float64{pt[0], pt[1]})
		}

		feature := map[string]interface{}{
			"type": "Feature",
			"properties": map[string]interface{}{
				"matched": true,
				"way_id":  wayID,
				"stroke":  randomColor(),
			},
			"geometry": map[string]interface{}{
				"type":        "LineString",
				"coordinates": lineCoords,
			},
		}
		features = append(features, feature)
	}

	outGeoJSON := map[string]interface{}{
		"type":     "FeatureCollection",
		"features": features,
	}

	outBytes, err := json.MarshalIndent(outGeoJSON, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Matched Ways FeatureCollection GeoJSON:\n%s\n", string(outBytes))
}
