package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"

	"kuanb/gosm-router/geom"
	"kuanb/gosm-router/osm"
	"kuanb/gosm-router/routing"
)

// Server holds the graph and matcher for handling requests
type Server struct {
	graph   *osm.OsmGraph
	matcher *routing.HMMMapMatcher
}

// randomColor generates a random hex color string
func randomColor() string {
	const letters = "0123456789ABCDEF"
	b := make([]byte, 7)
	b[0] = '#'
	for i := 1; i < 7; i++ {
		b[i] = letters[rand.Intn(16)]
	}
	return string(b)
}

// handleMatch processes a GeoJSON request and returns matched ways
func (s *Server) handleMatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	// Parse GeoJSON
	var fc geom.GeoJSONFeatureCollection
	if err := json.Unmarshal(body, &fc); err != nil {
		http.Error(w, "Invalid GeoJSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Extract coordinates
	var coords []routing.Coordinate
	for _, feature := range fc.Features {
		for _, coord := range feature.Geometry.Coordinates {
			if len(coord) >= 2 {
				coords = append(coords, routing.Coordinate{
					Lon: coord[0],
					Lat: coord[1],
				})
			}
		}
	}

	if len(coords) == 0 {
		http.Error(w, "No coordinates found in GeoJSON", http.StatusBadRequest)
		return
	}

	log.Printf("Processing match request with %d coordinates", len(coords))

	// Run map matching
	match := s.matcher.Match(coords)

	// Build response GeoJSON
	features := make([]interface{}, 0, len(match.MatchedWays))
	for _, wayID := range match.MatchedWays {
		way := s.graph.Ways[int64(wayID)]
		if way == nil || len(way.Geometry) < 2 {
			continue
		}

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

	response := map[string]interface{}{
		"type":       "FeatureCollection",
		"features":   features,
		"confidence": match.Confidence,
	}

	// Send response
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

func main() {
	pbfFile := flag.String("pbf", "example.osm.pbf", "PBF file name in data/ directory")
	flag.Parse()

	log.Println("gosm-router starting...")

	// Load graph at startup
	log.Printf("Loading graph from %s", fmt.Sprintf("./data/%s", *pbfFile))
	graph := osm.LoadOsmFile(fmt.Sprintf("./data/%s", *pbfFile))
	log.Printf("Loaded graph: %d nodes, %d ways", len(graph.Nodes), len(graph.Ways))

	// Create matcher
	matcher := routing.NewHMMMapMatcher(graph)

	// Create server
	server := &Server{
		graph:   graph,
		matcher: matcher,
	}

	// Register routes
	http.HandleFunc("/match", server.handleMatch)

	// Health check endpoint
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	// Start server
	addr := ":8080"
	log.Printf("Listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}
