package geom

import (
	"math"

	"github.com/tidwall/rtree"
)

// RTreeItem represents an item stored in the RTree
type RTreeItem struct {
	ID int64
}

// RTree wraps tidwall/rtree for spatial indexing of ways
type RTree struct {
	tree *rtree.RTreeG[RTreeItem]
}

// NewRTree creates a new RTree
func NewRTree() *RTree {
	return &RTree{
		tree: &rtree.RTreeG[RTreeItem]{},
	}
}

// Insert adds an item to the RTree with the given bounding box
func (r *RTree) Insert(id int64, minLon, minLat, maxLon, maxLat float64) {
	r.tree.Insert(
		[2]float64{minLon, minLat},
		[2]float64{maxLon, maxLat},
		RTreeItem{ID: id},
	)
}

// Search returns all item IDs whose bounding boxes intersect with the query bbox
func (r *RTree) Search(minLon, minLat, maxLon, maxLat float64) []int64 {
	result := make([]int64, 0)
	r.tree.Search(
		[2]float64{minLon, minLat},
		[2]float64{maxLon, maxLat},
		func(min, max [2]float64, item RTreeItem) bool {
			result = append(result, item.ID)
			return true // continue searching
		},
	)
	return result
}

// SearchNearPoint returns all item IDs within a distance (in meters) of a point
func (r *RTree) SearchNearPoint(lon, lat, distanceMeters float64) []int64 {
	// Convert distance to approximate degrees
	// Adjust for latitude
	latRad := lat * math.Pi / 180.0
	metersPerDegreeLon := EarthRadiusMeters * math.Pi / 180.0 * math.Cos(latRad)
	metersPerDegreeLat := EarthRadiusMeters * math.Pi / 180.0

	deltaLon := distanceMeters / metersPerDegreeLon
	deltaLat := distanceMeters / metersPerDegreeLat

	return r.Search(lon-deltaLon, lat-deltaLat, lon+deltaLon, lat+deltaLat)
}

// Size returns the number of items in the RTree
func (r *RTree) Size() int {
	return r.tree.Len()
}
