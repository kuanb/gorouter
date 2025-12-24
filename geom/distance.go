package geom

import "math"

const EarthRadiusMeters = 6371000.0

// GreatCircleDistance calculates the distance between two points in meters using the Haversine formula
func GreatCircleDistance(lon1, lat1, lon2, lat2 float64) float64 {
	toRad := func(deg float64) float64 { return deg * math.Pi / 180.0 }

	dLat := toRad(lat2 - lat1)
	dLon := toRad(lon2 - lon1)
	lat1Rad := toRad(lat1)
	lat2Rad := toRad(lat2)

	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(lat1Rad)*math.Cos(lat2Rad)*math.Sin(dLon/2)*math.Sin(dLon/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	return EarthRadiusMeters * c
}

// PointToSegmentDistance returns the shortest distance in meters from point p to the line segment ab
// Uses equirectangular projection (accurate for short distances)
func PointToSegmentDistance(pLon, pLat, aLon, aLat, bLon, bLat float64) float64 {
	toRad := func(deg float64) float64 { return deg * math.Pi / 180.0 }

	// Equirectangular projection locally around point a
	lat := toRad(aLat)
	cosLat := math.Cos(lat)
	ax := toRad(aLon) * cosLat * EarthRadiusMeters
	ay := toRad(aLat) * EarthRadiusMeters
	bx := toRad(bLon) * cosLat * EarthRadiusMeters
	by := toRad(bLat) * EarthRadiusMeters
	px := toRad(pLon) * cosLat * EarthRadiusMeters
	py := toRad(pLat) * EarthRadiusMeters

	// Project point p onto segment ab
	dx := bx - ax
	dy := by - ay
	if dx == 0 && dy == 0 {
		// a and b are the same point
		return math.Hypot(px-ax, py-ay)
	}
	t := ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)
	if t < 0 {
		return math.Hypot(px-ax, py-ay)
	} else if t > 1 {
		return math.Hypot(px-bx, py-by)
	}
	projx := ax + t*dx
	projy := ay + t*dy
	return math.Hypot(px-projx, py-projy)
}
