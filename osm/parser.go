package osm

import (
	"io"
	"log"
	"os"
	"runtime"

	"kuanb/gosm-router/geom"

	"github.com/paulmach/orb"
	"github.com/qedus/osmpbf"
)

func LoadOsmFile(filePath string) *OsmGraph {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	d := osmpbf.NewDecoder(f)

	// use more memory from the start, it is faster
	d.SetBufferSize(osmpbf.MaxBlobSize)

	// start decoding with several goroutines, it is faster
	err = d.Start(runtime.GOMAXPROCS(-1))
	if err != nil {
		log.Fatal(err)
	}

	var nc, wc, rc uint64
	nodes := make(map[int64]*OsmNode)
	ways := make(map[int64]*OsmWay)

	for {
		if v, err := d.Decode(); err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		} else {
			switch v := v.(type) {
			case *osmpbf.Node:
				nodes[v.ID] = &OsmNode{
					ID:  OsmNodeId(v.ID),
					Lat: v.Lat,
					Lon: v.Lon,
				}
				nc++
			case *osmpbf.Way:
				nodeIDs := make([]OsmNodeId, len(v.NodeIDs))
				for i, id := range v.NodeIDs {
					nodeIDs[i] = OsmNodeId(id)
				}
				ways[v.ID] = &OsmWay{
					ID:      OsmWayId(v.ID),
					Highway: v.Tags["highway"],
					Nodes:   nodeIDs,
				}
				wc++
			case *osmpbf.Relation:
				// we ignore relations for now
				rc++
			default:
				log.Fatalf("unknown type %T\n", v)
			}
		}
	}

	highwayTypesList := []string{
		"motorway",
		"motorway_link",
		"trunk",
		"trunk_link",
		"primary",
		"primary_link",
		"secondary",
		"secondary_link",
		"tertiary",
		"tertiary_link",
		"residential",
		"service",
		"living_street",
	}

	// Build set for highwayTypesList for fast lookup
	whitelistedHighways := make(map[string]struct{}, len(highwayTypesList))
	for _, hw := range highwayTypesList {
		whitelistedHighways[hw] = struct{}{}
	}

	// Remove ways not whitelisted in highwayTypesList
	filteredWays := make(map[int64]*OsmWay)
	usedNodeIDs := make(map[OsmNodeId]struct{})
	for id, way := range ways {
		if _, ok := whitelistedHighways[way.Highway]; ok {
			filteredWays[id] = way
			for _, nid := range way.Nodes {
				usedNodeIDs[nid] = struct{}{}
			}
		}
	}
	numWaysBefore := len(ways)
	numWaysAfter := len(filteredWays)
	ways = filteredWays
	log.Printf("Dropped %d ways (kept %d)", numWaysBefore-numWaysAfter, numWaysAfter)

	// Remove any nodes not used in the remaining ways
	filteredNodes := make(map[int64]*OsmNode)
	for id, node := range nodes {
		if _, ok := usedNodeIDs[OsmNodeId(id)]; ok {
			filteredNodes[id] = node
		}
	}
	origNodeCount := len(nodes)
	droppedNodes := origNodeCount - len(filteredNodes)
	log.Printf("Dropped %d nodes (kept %d)", droppedNodes, len(filteredNodes))
	nodes = filteredNodes

	// 1. Identify intersection nodes (those that appear in more than one way)
	nodeWayCount := make(map[OsmNodeId]int)
	for _, way := range ways {
		for _, nid := range way.Nodes {
			nodeWayCount[nid]++
		}
	}
	intersectionNodes := make(map[OsmNodeId]struct{})
	for nid, count := range nodeWayCount {
		if count > 1 {
			intersectionNodes[nid] = struct{}{}
		}
	}

	// 2. Build new set of nodes (only intersection nodes)
	resultNodes := make(map[int64]*OsmNode)
	for nid := range intersectionNodes {
		if n, ok := nodes[int64(nid)]; ok {
			resultNodes[int64(nid)] = n
		}
	}

	// 3. For each way, break it into segments between intersection nodes and create new 'ways'/edges
	resultWays := make(map[int64]*OsmWay)
	var newWayID int64 = 1
	for _, way := range ways {
		// gather indices of intersection nodes in this way
		var segStart int = -1
		var lastIntersection OsmNodeId
		for i, nid := range way.Nodes {
			_, isIntersection := intersectionNodes[nid]
			if isIntersection {
				if segStart == -1 {
					// First intersection node in way segment
					segStart = i
					lastIntersection = nid
				} else {
					// Next intersection: gather segment from lastIntersection to nid
					segment := way.Nodes[segStart : i+1]
					if len(segment) >= 2 {
						// Ensure both ends are intersection nodes
						geom := buildLineString(segment, nodes)
						newWay := &OsmWay{
							ID:       OsmWayId(newWayID),
							Nodes:    []OsmNodeId{lastIntersection, nid},
							Highway:  way.Highway,
							Geometry: geom,
						}
						resultWays[newWayID] = newWay
						newWayID++
					}
					segStart = i
					lastIntersection = nid
				}
			}
		}
	}

	// Build RTree spatial index for ways
	rtree := geom.NewRTree()
	for id, way := range resultWays {
		if len(way.Geometry) == 0 {
			continue
		}
		// Calculate bounding box for the way geometry
		minLon, minLat := way.Geometry[0][0], way.Geometry[0][1]
		maxLon, maxLat := minLon, minLat
		for _, pt := range way.Geometry {
			if pt[0] < minLon {
				minLon = pt[0]
			}
			if pt[0] > maxLon {
				maxLon = pt[0]
			}
			if pt[1] < minLat {
				minLat = pt[1]
			}
			if pt[1] > maxLat {
				maxLat = pt[1]
			}
		}
		rtree.Insert(id, minLon, minLat, maxLon, maxLat)
	}
	log.Printf("Built RTree with %d entries", rtree.Size())

	graph := &OsmGraph{
		Nodes: resultNodes,
		Ways:  resultWays,
		RTree: rtree,
	}
	return graph
}

// buildLineString creates a LineString geometry from a slice of node IDs
func buildLineString(nodeIDs []OsmNodeId, nodes map[int64]*OsmNode) orb.LineString {
	geom := make(orb.LineString, 0, len(nodeIDs))
	for _, nid := range nodeIDs {
		if node, ok := nodes[int64(nid)]; ok {
			geom = append(geom, orb.Point{node.Lon, node.Lat})
		}
	}
	return geom
}
