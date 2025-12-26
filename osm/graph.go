package osm

import (
	"kuanb/gosm-router/geom"

	"github.com/paulmach/orb"
)

type OsmWayId int64

type OsmNodeId int64

type OsmNode struct {
	ID  OsmNodeId
	Lat float64
	Lon float64
}

type OsmWay struct {
	ID           OsmWayId
	Nodes        []OsmNodeId
	Highway      string
	Geometry     orb.LineString
	LengthMeters float64 // Total length of the way in meters
}

func (w *OsmWay) MinDistanceToLonLat(lon, lat float64) float64 {
	if len(w.Geometry) == 0 {
		return -1
	}
	minDist := -1.0
	for i := 0; i < len(w.Geometry)-1; i++ {
		a := w.Geometry[i]
		b := w.Geometry[i+1]
		d := geom.PointToSegmentDistance(lon, lat, a[0], a[1], b[0], b[1])
		if minDist < 0 || d < minDist {
			minDist = d
		}
	}
	return minDist
}

type OsmGraph struct {
	Nodes map[int64]*OsmNode
	Ways  map[int64]*OsmWay
	RTree *geom.RTree
}
