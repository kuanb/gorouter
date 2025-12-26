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
	MatchedWays []osm.OsmWayId // sequence of matched way IDs (one per probe, may have duplicates)
	FinalPath   []osm.OsmWayId // deduplicated path with intermediate edges filled in
	Confidence  float64        // overall confidence score (0-1)
}

// HMMMapMatcher performs map matching using a Hidden Markov Model
type HMMMapMatcher struct {
	Graph            *osm.OsmGraph
	SigmaZ           float64 // GPS measurement noise (meters), typically 4.07
	Beta             float64 // transition probability parameter, typically 3.0
	MaxCandidateDist float64 // maximum distance to consider a candidate (meters)
	UTurnPenalty     float64 // penalty multiplier for u-turn patterns (0-1, lower = more penalty)
}

// NewHMMMapMatcher creates a new HMM map matcher with default parameters
func NewHMMMapMatcher(graph *osm.OsmGraph) *HMMMapMatcher {
	return &HMMMapMatcher{
		Graph:            graph,
		SigmaZ:           4.07,  // typical GPS noise
		Beta:             3.0,   // transition parameter
		MaxCandidateDist: 35.0,  // max 35m from GPS point
		UTurnPenalty:     1e-50, // extremely severe penalty for u-turns
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

	// Step 3: Compute final path (deduplicated with intermediate edges filled in)
	finalPath := m.computeFinalPath(path)

	return MatchResult{
		MatchedWays: path,
		FinalPath:   finalPath,
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

// roadClassRank returns a numeric ranking for highway types (higher = more important road)
// This is used to penalize transitions that "dip" to lower road classes
func roadClassRank(highway string) int {
	switch highway {
	case "motorway":
		return 100
	case "motorway_link":
		return 95
	case "trunk":
		return 90
	case "trunk_link":
		return 85
	case "primary":
		return 80
	case "primary_link":
		return 75
	case "secondary":
		return 70
	case "secondary_link":
		return 65
	case "tertiary":
		return 60
	case "tertiary_link":
		return 55
	case "residential":
		return 40
	case "living_street":
		return 30
	case "service":
		return 20
	default:
		return 10
	}
}

// calculateTurnAngle calculates the turn angle in degrees between two connected edges
// Returns 180 for going straight, 90 for perpendicular turn, 0 for u-turn
// Returns -1 if the edges don't share a node or geometry is invalid
func (m *HMMMapMatcher) calculateTurnAngle(fromWay, toWay *osm.OsmWay) float64 {
	if fromWay == nil || toWay == nil {
		return -1
	}
	if len(fromWay.Geometry) < 2 || len(toWay.Geometry) < 2 {
		return -1
	}

	// Find the shared node and determine which end of each edge connects
	fromNodes := fromWay.Nodes
	toNodes := toWay.Nodes

	// Determine connection point and get direction vectors
	var approachVec, departVec [2]float64
	foundConnection := false

	// Check all four possible connection scenarios:
	// 1. fromWay end -> toWay start
	// 2. fromWay end -> toWay end
	// 3. fromWay start -> toWay start
	// 4. fromWay start -> toWay end

	fromGeom := fromWay.Geometry
	toGeom := toWay.Geometry

	if fromNodes[len(fromNodes)-1] == toNodes[0] {
		// fromWay ends at toWay start - normal forward connection
		// Approach: second-to-last point toward last point of fromWay
		approachVec = [2]float64{
			fromGeom[len(fromGeom)-1][0] - fromGeom[len(fromGeom)-2][0],
			fromGeom[len(fromGeom)-1][1] - fromGeom[len(fromGeom)-2][1],
		}
		// Depart: first point toward second point of toWay
		departVec = [2]float64{
			toGeom[1][0] - toGeom[0][0],
			toGeom[1][1] - toGeom[0][1],
		}
		foundConnection = true
	} else if fromNodes[len(fromNodes)-1] == toNodes[len(toNodes)-1] {
		// fromWay ends at toWay end - toWay is traversed backwards
		approachVec = [2]float64{
			fromGeom[len(fromGeom)-1][0] - fromGeom[len(fromGeom)-2][0],
			fromGeom[len(fromGeom)-1][1] - fromGeom[len(fromGeom)-2][1],
		}
		// Depart: last point toward second-to-last point of toWay (reverse direction)
		departVec = [2]float64{
			toGeom[len(toGeom)-2][0] - toGeom[len(toGeom)-1][0],
			toGeom[len(toGeom)-2][1] - toGeom[len(toGeom)-1][1],
		}
		foundConnection = true
	} else if fromNodes[0] == toNodes[0] {
		// fromWay start connects to toWay start - fromWay is traversed backwards
		approachVec = [2]float64{
			fromGeom[0][0] - fromGeom[1][0],
			fromGeom[0][1] - fromGeom[1][1],
		}
		departVec = [2]float64{
			toGeom[1][0] - toGeom[0][0],
			toGeom[1][1] - toGeom[0][1],
		}
		foundConnection = true
	} else if fromNodes[0] == toNodes[len(toNodes)-1] {
		// fromWay start connects to toWay end - both traversed backwards
		approachVec = [2]float64{
			fromGeom[0][0] - fromGeom[1][0],
			fromGeom[0][1] - fromGeom[1][1],
		}
		departVec = [2]float64{
			toGeom[len(toGeom)-2][0] - toGeom[len(toGeom)-1][0],
			toGeom[len(toGeom)-2][1] - toGeom[len(toGeom)-1][1],
		}
		foundConnection = true
	}

	if !foundConnection {
		return -1
	}

	// Calculate angle between vectors using dot product
	// cos(θ) = (a · b) / (|a| |b|)
	dot := approachVec[0]*departVec[0] + approachVec[1]*departVec[1]
	magA := math.Sqrt(approachVec[0]*approachVec[0] + approachVec[1]*approachVec[1])
	magB := math.Sqrt(departVec[0]*departVec[0] + departVec[1]*departVec[1])

	if magA == 0 || magB == 0 {
		return -1
	}

	cosAngle := dot / (magA * magB)
	// Clamp to [-1, 1] to handle floating point errors
	cosAngle = math.Max(-1, math.Min(1, cosAngle))

	// Convert to degrees
	angleRad := math.Acos(cosAngle)
	angleDeg := angleRad * 180 / math.Pi

	return angleDeg
}

// turnAnglePenalty returns a penalty factor (0-1) based on turn angle
// 180° (straight) = 1.0 (no penalty)
// 90° (perpendicular) = moderate penalty
// 0° (u-turn) = severe penalty
func turnAnglePenalty(angleDeg float64) float64 {
	if angleDeg < 0 {
		return 1.0 // unknown angle, no penalty
	}

	// Normalize angle to 0-180 range
	if angleDeg > 180 {
		angleDeg = 360 - angleDeg
	}

	// Use a smooth function that penalizes sharper turns more heavily
	// At 180° (straight): penalty = 1.0
	// At 90° (perpendicular): penalty ≈ 0.7
	// At 45° (sharp turn): penalty ≈ 0.4
	// At 0° (u-turn): penalty ≈ 0.1
	penalty := 0.1 + 0.9*(angleDeg/180.0)*(angleDeg/180.0)

	return penalty
}

// transitionProbability calculates P(state_t | state_{t-1})
// Based on the difference between great circle distance and route distance
// Also considers road hierarchy and turn angle penalties
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

	var baseProb float64
	if connected {
		// Connected roads get higher probability
		baseProb = math.Exp(-gcDist / (m.Beta * 100))

		// Apply turn angle penalty - sharper turns are less likely
		turnAngle := m.calculateTurnAngle(fromWayPtr, toWayPtr)
		baseProb *= turnAnglePenalty(turnAngle)
	} else {
		// Non-connected roads get lower probability based on distance
		baseProb = math.Exp(-gcDist/m.Beta) * 0.1
	}

	// Apply road hierarchy penalty for transitioning to lower road classes
	fromRank := roadClassRank(fromWayPtr.Highway)
	toRank := roadClassRank(toWayPtr.Highway)

	if toRank < fromRank {
		// Transitioning to a lower road class - apply penalty
		// The penalty is proportional to how much lower the target road is
		rankDiff := fromRank - toRank
		// Penalty ranges from 0.9 (small drop) to 0.1 (large drop like primary->service)
		penalty := math.Max(0.1, 1.0-float64(rankDiff)/100.0)
		baseProb *= penalty
	}

	return baseProb
}

// viterbi runs a second-order Viterbi algorithm to find the most likely path
// It tracks the previous-previous state to detect and penalize u-turn patterns (A→B→A)
func (m *HMMMapMatcher) viterbi(coords []Coordinate, candidates [][]Candidate) ([]osm.OsmWayId, float64) {
	n := len(coords)
	if n == 0 {
		return nil, 0
	}

	// V[t][i] = probability of most likely path ending in candidate i at time t
	V := make([][]float64, n)
	// backpointer[t][i] = index of previous candidate in most likely path
	backpointer := make([][]int, n)
	// prevWayAtBest[t][i] = the way ID at t-2 for the best path ending at candidate i at time t
	// Used to detect u-turn patterns
	prevPrevWay := make([][]osm.OsmWayId, n)

	for t := 0; t < n; t++ {
		V[t] = make([]float64, len(candidates[t]))
		backpointer[t] = make([]int, len(candidates[t]))
		prevPrevWay[t] = make([]osm.OsmWayId, len(candidates[t]))
	}

	// Initialize: first observation
	for i, cand := range candidates[0] {
		V[0][i] = math.Log(m.emissionProbability(cand.Distance) + 1e-10)
		backpointer[0][i] = -1
		prevPrevWay[0][i] = -1 // no previous-previous for first observation
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
			bestPrevPrevWay := osm.OsmWayId(-1)

			for i, prevCand := range candidates[t-1] {
				transProb := m.transitionProbability(prevCand.WayID, currCand.WayID, gcDist)

				// Check for u-turn pattern: if current way equals the way from t-2,
				// and it's different from t-1, this is a u-turn (A→B→A)
				if t >= 2 {
					wayAtTMinus2 := prevPrevWay[t-1][i]
					if wayAtTMinus2 > 0 && wayAtTMinus2 == currCand.WayID && prevCand.WayID != currCand.WayID {
						// This is a u-turn: went from A to B and now back to A
						transProb *= m.UTurnPenalty

						// Additional penalty if this is a "dip" to a lower road class
						// Pattern: primary -> residential -> primary is very suspicious
						prevWayPtr := m.Graph.Ways[int64(prevCand.WayID)]
						currWayPtr := m.Graph.Ways[int64(currCand.WayID)]
						if prevWayPtr != nil && currWayPtr != nil {
							prevRank := roadClassRank(prevWayPtr.Highway)
							currRank := roadClassRank(currWayPtr.Highway)
							if prevRank < currRank {
								// The intermediate edge (B) is lower class than the bookends (A)
								// This is a classic false match at an intersection
								rankDiff := currRank - prevRank
								// Apply additional penalty based on how much lower B is
								dipPenalty := math.Max(0.01, 1.0-float64(rankDiff)/50.0)
								transProb *= dipPenalty
							}
						}
					}
				}

				prob := V[t-1][i] + math.Log(transProb+1e-10)

				if prob > maxProb {
					maxProb = prob
					maxIdx = i
					bestPrevPrevWay = prevCand.WayID // the way at t-1 becomes t-2 for next iteration
				}
			}

			emitProb := m.emissionProbability(currCand.Distance)
			V[t][j] = maxProb + math.Log(emitProb+1e-10)
			backpointer[t][j] = maxIdx
			prevPrevWay[t][j] = bestPrevPrevWay
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

	// Post-process: detect and fix isolated detour spikes
	path = m.removeIsolatedDetours(path)

	// Post-process: remove low-coverage edges that look like intersection noise
	path = m.removeLowCoverageDetours(path)

	// Calculate confidence score (normalized probability)
	// Convert log probability to 0-1 scale
	confidence := calculateConfidence(V, n, maxIdx)

	return path, confidence
}

// removeIsolatedDetours detects patterns where a single observation is on a different
// edge while surrounding observations are on the same edge, and corrects them.
// Pattern: [A, A, A, B, A, A] -> [A, A, A, A, A, A] when B is an isolated spike
func (m *HMMMapMatcher) removeIsolatedDetours(path []osm.OsmWayId) []osm.OsmWayId {
	if len(path) < 3 {
		return path
	}

	result := make([]osm.OsmWayId, len(path))
	copy(result, path)

	// Detect isolated spikes: single observations on a different edge
	for i := 1; i < len(path)-1; i++ {
		prev := path[i-1]
		curr := path[i]
		next := path[i+1]

		// If current is different from both neighbors, and neighbors are the same,
		// this is likely an isolated spike
		if curr != prev && curr != next && prev == next {
			// Check if the detour edge is connected to the main edge (could be an intersection)
			currWay := m.Graph.Ways[int64(curr)]
			prevWay := m.Graph.Ways[int64(prev)]

			if currWay != nil && prevWay != nil {
				// Check if they share a node (intersecting roads)
				sharesNode := false
				for _, n1 := range currWay.Nodes {
					for _, n2 := range prevWay.Nodes {
						if n1 == n2 {
							sharesNode = true
							break
						}
					}
					if sharesNode {
						break
					}
				}

				// If they share a node, this is likely a false match to an intersecting road
				// Replace the spike with the surrounding edge
				if sharesNode {
					result[i] = prev
				}
			}
		}
	}

	// Also detect short detour sequences: [A, A, B, B, A, A] -> potential u-turn
	// Look for pattern where we leave an edge and return within 2-3 observations
	for i := 0; i < len(result)-3; i++ {
		if result[i] == result[i+3] && result[i] != result[i+1] && result[i+1] == result[i+2] {
			// Pattern: A, B, B, A - a 2-observation detour
			// Check if B is connected to A (intersection)
			edgeA := result[i]
			edgeB := result[i+1]

			wayA := m.Graph.Ways[int64(edgeA)]
			wayB := m.Graph.Ways[int64(edgeB)]

			if wayA != nil && wayB != nil {
				sharesNode := false
				for _, n1 := range wayA.Nodes {
					for _, n2 := range wayB.Nodes {
						if n1 == n2 {
							sharesNode = true
							break
						}
					}
					if sharesNode {
						break
					}
				}

				// If connected and it's a short detour, likely a false match
				if sharesNode {
					result[i+1] = edgeA
					result[i+2] = edgeA
				}
			}
		}
	}

	return result
}

// edgeRun represents a contiguous run of probes matched to the same edge
type edgeRun struct {
	wayID        osm.OsmWayId
	start        int     // start index in original path
	end          int     // end index (exclusive) in original path
	count        int     // number of probes in this run
	lengthMeters float64 // length of the edge in meters
	coverage     float64 // probes per meter (count / lengthMeters)
}

// removeLowCoverageDetours identifies edges with low probe coverage that are sandwiched
// between edges with much higher coverage, suggesting intersection noise rather than
// actual traversal. Pattern: [A x 10, B x 2, A x 8] -> likely B is noise at intersection
func (m *HMMMapMatcher) removeLowCoverageDetours(path []osm.OsmWayId) []osm.OsmWayId {
	if len(path) < 3 {
		return path
	}

	// Step 1: Identify contiguous runs of the same edge
	runs := m.identifyEdgeRuns(path)

	if len(runs) < 3 {
		return path
	}

	// Step 2: Identify low-coverage runs that should be collapsed
	// Look for pattern: high-coverage A, low-coverage B (connected to A), high-coverage A
	result := make([]osm.OsmWayId, len(path))
	copy(result, path)

	// Multiple passes to handle cascading fixes
	for pass := 0; pass < 3; pass++ {
		runs = m.identifyEdgeRuns(result)
		if len(runs) < 3 {
			break
		}

		changed := false
		for i := 1; i < len(runs)-1; i++ {
			prevRun := runs[i-1]
			currRun := runs[i]
			nextRun := runs[i+1]

			// Check if this is a potential false detour:
			// - Current run has low coverage density (few probes per meter)
			// - Surrounded by runs with higher coverage density
			// - Surrounding runs are the same edge (suggests u-turn pattern)
			// - Current edge is connected to surrounding edge (intersection)

			// Condition 1: Surrounding edges are the same
			if prevRun.wayID != nextRun.wayID {
				continue
			}

			// Condition 2: Current run has significantly less coverage density (probes per meter)
			surroundingAvgCoverage := (prevRun.coverage + nextRun.coverage) / 2.0
			coverageRatio := currRun.coverage / surroundingAvgCoverage

			// If current edge has less than 50% of surrounding coverage density, it's suspicious
			if coverageRatio >= 0.5 {
				continue
			}

			// Condition 3: Current edge must be connected to surrounding edge (intersection)
			if !m.waysAreConnected(currRun.wayID, prevRun.wayID) {
				continue
			}

			// This looks like intersection noise - collapse current run to surrounding edge
			for j := currRun.start; j < currRun.end; j++ {
				result[j] = prevRun.wayID
			}
			changed = true
		}

		if !changed {
			break
		}
	}

	// Step 2b: Handle transition case: [A x many, B x few, C x many]
	// where A != C but B is a low-coverage edge at the intersection
	// B should be assigned to either A or C based on connectivity
	for pass := 0; pass < 3; pass++ {
		runs = m.identifyEdgeRuns(result)
		if len(runs) < 3 {
			break
		}

		changed := false
		for i := 1; i < len(runs)-1; i++ {
			prevRun := runs[i-1]
			currRun := runs[i]
			nextRun := runs[i+1]

			// Skip if surrounding edges are the same (handled above)
			if prevRun.wayID == nextRun.wayID {
				continue
			}

			// Check if current run has very low coverage density compared to neighbors
			minNeighborCoverage := prevRun.coverage
			if nextRun.coverage < minNeighborCoverage {
				minNeighborCoverage = nextRun.coverage
			}

			// Current edge must have significantly lower coverage density than both neighbors
			// Use relative threshold: current coverage must be less than 50% of minimum neighbor coverage
			if currRun.coverage >= minNeighborCoverage*0.5 {
				continue
			}

			// Also require current run to have few total probes (to avoid removing legitimate traversals)
			if currRun.count > 4 {
				continue
			}

			// Check connectivity: B should be connected to both A and C for this to be intersection noise
			connectedToPrev := m.waysAreConnected(currRun.wayID, prevRun.wayID)
			connectedToNext := m.waysAreConnected(currRun.wayID, nextRun.wayID)

			if !connectedToPrev && !connectedToNext {
				continue
			}

			// Also verify A and C are connected (otherwise B might be a legitimate bridge)
			if !m.waysAreConnected(prevRun.wayID, nextRun.wayID) {
				continue
			}

			// B is intersection noise - assign to the edge it's more connected to,
			// or to the one with higher coverage density
			var targetWayID osm.OsmWayId
			if connectedToPrev && connectedToNext {
				// Connected to both - choose the one with higher coverage density
				if prevRun.coverage >= nextRun.coverage {
					targetWayID = prevRun.wayID
				} else {
					targetWayID = nextRun.wayID
				}
			} else if connectedToPrev {
				targetWayID = prevRun.wayID
			} else {
				targetWayID = nextRun.wayID
			}

			for j := currRun.start; j < currRun.end; j++ {
				result[j] = targetWayID
			}
			changed = true
		}

		if !changed {
			break
		}
	}

	// Step 3: Handle longer low-coverage sequences (not just single edges)
	// Look for: [A x many, B x few, C x few, A x many] where B and C are short detours
	runs = m.identifyEdgeRuns(result)
	for i := 0; i < len(runs)-1; i++ {
		// Find runs that return to a previous edge
		for j := i + 2; j < len(runs) && j <= i+4; j++ {
			if runs[i].wayID == runs[j].wayID {
				// Found a return pattern: runs[i] ... runs[j] have same wayID
				// Check if intermediate runs have low average coverage density
				intermediateTotalCoverage := 0.0
				intermediateRunCount := 0
				for k := i + 1; k < j; k++ {
					intermediateTotalCoverage += runs[k].coverage
					intermediateRunCount++
				}
				intermediateAvgCoverage := intermediateTotalCoverage / float64(intermediateRunCount)

				bookendAvgCoverage := (runs[i].coverage + runs[j].coverage) / 2.0
				if intermediateAvgCoverage < bookendAvgCoverage*0.6 {
					// Low coverage density detour sequence - check if connected
					allConnected := true
					for k := i + 1; k < j && allConnected; k++ {
						if !m.waysAreConnected(runs[k].wayID, runs[i].wayID) {
							allConnected = false
						}
					}

					if allConnected {
						// Collapse all intermediate runs to the bookend edge
						for k := i + 1; k < j; k++ {
							for idx := runs[k].start; idx < runs[k].end; idx++ {
								result[idx] = runs[i].wayID
							}
						}
					}
				}
				break // Only handle first return pattern from this starting point
			}
		}
	}

	return result
}

// identifyEdgeRuns finds contiguous sequences of the same edge in the path
// and calculates coverage density (probes per meter) for each run
func (m *HMMMapMatcher) identifyEdgeRuns(path []osm.OsmWayId) []edgeRun {
	if len(path) == 0 {
		return nil
	}

	// Helper to get way length, with fallback to avoid division by zero
	getWayLength := func(wayID osm.OsmWayId) float64 {
		way := m.Graph.Ways[int64(wayID)]
		if way != nil && way.LengthMeters > 0 {
			return way.LengthMeters
		}
		return 1.0 // fallback to 1 meter to avoid division by zero
	}

	runs := make([]edgeRun, 0)
	currentRun := edgeRun{
		wayID:        path[0],
		start:        0,
		end:          1,
		count:        1,
		lengthMeters: getWayLength(path[0]),
	}

	for i := 1; i < len(path); i++ {
		if path[i] == currentRun.wayID {
			currentRun.end = i + 1
			currentRun.count++
		} else {
			// Finalize current run with coverage calculation
			currentRun.coverage = float64(currentRun.count) / currentRun.lengthMeters
			runs = append(runs, currentRun)
			currentRun = edgeRun{
				wayID:        path[i],
				start:        i,
				end:          i + 1,
				count:        1,
				lengthMeters: getWayLength(path[i]),
			}
		}
	}
	// Finalize last run
	currentRun.coverage = float64(currentRun.count) / currentRun.lengthMeters
	runs = append(runs, currentRun)

	return runs
}

// computeFinalPath takes the per-probe matched ways and creates a clean path:
// 1. Removes consecutive duplicate edges
// 2. Fills in intermediate edges between non-adjacent matched ways using BFS
func (m *HMMMapMatcher) computeFinalPath(matchedWays []osm.OsmWayId) []osm.OsmWayId {
	if len(matchedWays) == 0 {
		return nil
	}

	// Step 1: Deduplicate consecutive matches
	deduped := make([]osm.OsmWayId, 0, len(matchedWays))
	var lastWay osm.OsmWayId = -1
	for _, wayID := range matchedWays {
		if wayID != lastWay {
			deduped = append(deduped, wayID)
			lastWay = wayID
		}
	}

	if len(deduped) <= 1 {
		return deduped
	}

	// Step 2: Fill in intermediate edges between non-adjacent ways
	finalPath := make([]osm.OsmWayId, 0, len(deduped)*2)
	finalPath = append(finalPath, deduped[0])

	for i := 1; i < len(deduped); i++ {
		fromWay := deduped[i-1]
		toWay := deduped[i]

		// Check if ways are directly connected
		if m.waysAreConnected(fromWay, toWay) {
			finalPath = append(finalPath, toWay)
		} else {
			// Find intermediate path using BFS
			intermediatePath := m.findPathBetweenWays(fromWay, toWay)
			if len(intermediatePath) > 0 {
				// intermediatePath includes fromWay, so skip it
				finalPath = append(finalPath, intermediatePath[1:]...)
			} else {
				// No path found, just add the target way
				finalPath = append(finalPath, toWay)
			}
		}
	}

	return finalPath
}

// waysAreConnected checks if two ways share a node
func (m *HMMMapMatcher) waysAreConnected(wayA, wayB osm.OsmWayId) bool {
	wayAPtr := m.Graph.Ways[int64(wayA)]
	wayBPtr := m.Graph.Ways[int64(wayB)]

	if wayAPtr == nil || wayBPtr == nil {
		return false
	}

	for _, n1 := range wayAPtr.Nodes {
		for _, n2 := range wayBPtr.Nodes {
			if n1 == n2 {
				return true
			}
		}
	}
	return false
}

// findPathBetweenWays uses BFS to find the shortest path (in terms of ways) between two ways
// Returns the path including both fromWay and toWay, or empty slice if no path found
func (m *HMMMapMatcher) findPathBetweenWays(fromWay, toWay osm.OsmWayId) []osm.OsmWayId {
	if fromWay == toWay {
		return []osm.OsmWayId{fromWay}
	}

	// Build adjacency: node -> list of ways that contain this node
	nodeToWays := m.buildNodeToWaysMap()

	// BFS state
	type bfsState struct {
		wayID osm.OsmWayId
		path  []osm.OsmWayId
	}

	visited := make(map[osm.OsmWayId]bool)
	queue := []bfsState{{wayID: fromWay, path: []osm.OsmWayId{fromWay}}}
	visited[fromWay] = true

	// Limit search depth to prevent infinite loops
	maxDepth := 10

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		if len(current.path) > maxDepth {
			continue
		}

		// Get all adjacent ways
		currentWay := m.Graph.Ways[int64(current.wayID)]
		if currentWay == nil {
			continue
		}

		for _, nodeID := range currentWay.Nodes {
			adjacentWays := nodeToWays[nodeID]
			for _, adjWayID := range adjacentWays {
				if adjWayID == toWay {
					// Found the target
					return append(current.path, toWay)
				}

				if !visited[adjWayID] {
					visited[adjWayID] = true
					newPath := make([]osm.OsmWayId, len(current.path)+1)
					copy(newPath, current.path)
					newPath[len(current.path)] = adjWayID
					queue = append(queue, bfsState{wayID: adjWayID, path: newPath})
				}
			}
		}
	}

	// No path found
	return nil
}

// buildNodeToWaysMap creates a map from node ID to list of ways containing that node
func (m *HMMMapMatcher) buildNodeToWaysMap() map[osm.OsmNodeId][]osm.OsmWayId {
	nodeToWays := make(map[osm.OsmNodeId][]osm.OsmWayId)

	for _, way := range m.Graph.Ways {
		for _, nodeID := range way.Nodes {
			nodeToWays[nodeID] = append(nodeToWays[nodeID], way.ID)
		}
	}

	return nodeToWays
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
