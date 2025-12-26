package routing

import (
	"math"

	"kuanb/gosm-router/geom"
	"kuanb/gosm-router/osm"
)

// Coordinate represents a GPS observation point
type Coordinate struct {
	Lon float64
	Lat float64
}

// Candidate represents a potential road segment match for a GPS point
type Candidate struct {
	WayID    osm.OsmWayId
	Distance float64 // distance from GPS point to road segment in meters
}

// MatchResult represents the output of the HMM map matching
type MatchResult struct {
	MatchedWays []osm.OsmWayId // sequence of matched way IDs
	Confidence  float64        // overall confidence score (0-1)
}

// HMMMapMatcher performs map matching using a Hidden Markov Model
type HMMMapMatcher struct {
	Graph            *osm.OsmGraph
	SigmaZ           float64 // GPS measurement noise (meters), typically 4.07
	Beta             float64 // transition probability parameter, typically 3.0
	MaxCandidateDist float64 // maximum distance to consider a candidate (meters)
}

// NewHMMMapMatcher creates a new HMM map matcher with default parameters
func NewHMMMapMatcher(graph *osm.OsmGraph) *HMMMapMatcher {
	return &HMMMapMatcher{
		Graph:            graph,
		SigmaZ:           4.07, // typical GPS noise
		Beta:             3.0,  // transition parameter
		MaxCandidateDist: 35.0, // max 35m from GPS point
	}
}

// Match performs HMM map matching on a sequence of coordinates
func (m *HMMMapMatcher) Match(coords []Coordinate) MatchResult {
	if len(coords) == 0 {
		return MatchResult{MatchedWays: nil, Confidence: 0}
	}

	// Step 1: Find candidates for each observation
	candidates := m.findCandidates(coords)

	// Check if any observation has no candidates
	for i, c := range candidates {
		if len(c) == 0 {
			// No candidates for this point - return empty result
			return MatchResult{
				MatchedWays: nil,
				Confidence:  0,
			}
		}
		_ = i
	}

	// Step 2: Run Viterbi algorithm
	path, confidence := m.viterbi(coords, candidates)

	return MatchResult{
		MatchedWays: path,
		Confidence:  confidence,
	}
}

// findCandidates finds all road segment candidates within MaxCandidateDist for each coordinate
func (m *HMMMapMatcher) findCandidates(coords []Coordinate) [][]Candidate {
	candidates := make([][]Candidate, len(coords))

	for i, coord := range coords {
		candidates[i] = make([]Candidate, 0)

		// Use RTree for fast spatial lookup if available
		if m.Graph.RTree != nil {
			wayIDs := m.Graph.RTree.SearchNearPoint(coord.Lon, coord.Lat, m.MaxCandidateDist)
			for _, wayID := range wayIDs {
				way := m.Graph.Ways[wayID]
				if way == nil {
					continue
				}
				dist := way.MinDistanceToLonLat(coord.Lon, coord.Lat)
				if dist >= 0 && dist <= m.MaxCandidateDist {
					candidates[i] = append(candidates[i], Candidate{
						WayID:    way.ID,
						Distance: dist,
					})
				}
			}
		} else {
			// Fallback to brute force search
			for _, way := range m.Graph.Ways {
				dist := way.MinDistanceToLonLat(coord.Lon, coord.Lat)
				if dist >= 0 && dist <= m.MaxCandidateDist {
					candidates[i] = append(candidates[i], Candidate{
						WayID:    way.ID,
						Distance: dist,
					})
				}
			}
		}
	}

	return candidates
}

// emissionProbability calculates P(observation | state) using Gaussian distribution
func (m *HMMMapMatcher) emissionProbability(distance float64) float64 {
	// Gaussian probability: exp(-0.5 * (d/sigma)^2)
	return math.Exp(-0.5 * math.Pow(distance/m.SigmaZ, 2))
}

// transitionProbability calculates P(state_t | state_{t-1})
// Based on the difference between great circle distance and route distance
func (m *HMMMapMatcher) transitionProbability(fromWay, toWay osm.OsmWayId, gcDist float64) float64 {
	// For simplicity, we use an exponential distribution based on great circle distance
	// In a full implementation, you'd compute actual route distance
	// P(transition) = exp(-|routeDist - gcDist| / beta)

	// If same way, high probability
	if fromWay == toWay {
		return 1.0
	}

	// Check if ways are connected (share a node)
	fromWayPtr := m.Graph.Ways[int64(fromWay)]
	toWayPtr := m.Graph.Ways[int64(toWay)]

	if fromWayPtr == nil || toWayPtr == nil {
		return 0.001
	}

	// Check connectivity
	connected := false
	for _, n1 := range fromWayPtr.Nodes {
		for _, n2 := range toWayPtr.Nodes {
			if n1 == n2 {
				connected = true
				break
			}
		}
		if connected {
			break
		}
	}

	if connected {
		// Connected roads get higher probability
		return math.Exp(-gcDist / (m.Beta * 100))
	}

	// Non-connected roads get lower probability based on distance
	return math.Exp(-gcDist/m.Beta) * 0.1
}

// viterbi runs the Viterbi algorithm to find the most likely path
func (m *HMMMapMatcher) viterbi(coords []Coordinate, candidates [][]Candidate) ([]osm.OsmWayId, float64) {
	n := len(coords)
	if n == 0 {
		return nil, 0
	}

	// V[t][i] = probability of most likely path ending in candidate i at time t
	V := make([][]float64, n)
	// backpointer[t][i] = index of previous candidate in most likely path
	backpointer := make([][]int, n)

	for t := 0; t < n; t++ {
		V[t] = make([]float64, len(candidates[t]))
		backpointer[t] = make([]int, len(candidates[t]))
	}

	// Initialize: first observation
	for i, cand := range candidates[0] {
		V[0][i] = math.Log(m.emissionProbability(cand.Distance) + 1e-10)
		backpointer[0][i] = -1
	}

	// Recursion
	for t := 1; t < n; t++ {
		gcDist := geom.GreatCircleDistance(
			coords[t-1].Lon, coords[t-1].Lat,
			coords[t].Lon, coords[t].Lat,
		)

		for j, currCand := range candidates[t] {
			maxProb := math.Inf(-1)
			maxIdx := 0

			for i, prevCand := range candidates[t-1] {
				transProb := m.transitionProbability(prevCand.WayID, currCand.WayID, gcDist)
				prob := V[t-1][i] + math.Log(transProb+1e-10)

				if prob > maxProb {
					maxProb = prob
					maxIdx = i
				}
			}

			emitProb := m.emissionProbability(currCand.Distance)
			V[t][j] = maxProb + math.Log(emitProb+1e-10)
			backpointer[t][j] = maxIdx
		}
	}

	// Find best final state
	maxProb := math.Inf(-1)
	maxIdx := 0
	for i, prob := range V[n-1] {
		if prob > maxProb {
			maxProb = prob
			maxIdx = i
		}
	}

	// Backtrack to find path
	path := make([]osm.OsmWayId, n)
	idx := maxIdx
	for t := n - 1; t >= 0; t-- {
		path[t] = candidates[t][idx].WayID
		idx = backpointer[t][idx]
	}

	// Calculate confidence score (normalized probability)
	// Convert log probability to 0-1 scale
	confidence := calculateConfidence(V, n, maxIdx)

	return path, confidence
}

// calculateConfidence converts the Viterbi probabilities to a confidence score
func calculateConfidence(V [][]float64, n int, bestIdx int) float64 {
	if n == 0 {
		return 0
	}

	// Use the ratio of best path probability to average probability
	bestLogProb := V[n-1][bestIdx]

	// Calculate average log probability of all paths at final step
	sumExp := 0.0
	for _, logP := range V[n-1] {
		sumExp += math.Exp(logP - bestLogProb) // normalize to prevent overflow
	}
	avgLogProb := bestLogProb + math.Log(sumExp/float64(len(V[n-1])))

	// Confidence based on how much better best path is than average
	diff := bestLogProb - avgLogProb
	confidence := 1.0 - math.Exp(-diff)

	// Clamp to [0, 1]
	if confidence < 0 {
		confidence = 0
	}
	if confidence > 1 {
		confidence = 1
	}

	return confidence
}
