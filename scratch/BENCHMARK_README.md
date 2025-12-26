# Map Matching Benchmark Tool

Concurrent benchmark tool for testing map matching API performance with synthetic routes.

## Features

- **Concurrent requests**: Configurable batch size (default: 50 concurrent)
- **Performance metrics**: Response times, throughput, percentiles (P95, P99)
- **Error tracking**: Detailed error breakdown and status codes
- **Progress monitoring**: Real-time progress updates during execution
- **JSON output**: Detailed results saved for analysis

## Installation

```bash
pip install aiohttp
```

## Quick Start

```bash
# 1. Generate synthetic routes (if you haven't already)
python generate_synthetic_routes.py your_area.osm.pbf -n 1000 -o test_routes.geojsonl

# 2. Start your map matching server (in another terminal)
./your_map_matcher --port 8080

# 3. Run the benchmark
python benchmark_mapmatch.py test_routes.geojsonl
```

## Usage Examples

### Basic Usage

Test with all routes in a file:

```bash
python benchmark_mapmatch.py synthetic_routes.geojsonl
```

### Custom API URL

```bash
python benchmark_mapmatch.py synthetic_routes.geojsonl --api-url http://localhost:9000/match
```

### Adjust Concurrency

Test with 100 concurrent requests:

```bash
python benchmark_mapmatch.py synthetic_routes.geojsonl --batch-size 100
```

### Limit Number of Routes

Test with only the first 100 routes:

```bash
python benchmark_mapmatch.py synthetic_routes.geojsonl -n 100
```

### Complete Example

```bash
python benchmark_mapmatch.py test_routes.geojsonl \
    --api-url http://localhost:8080/match \
    --batch-size 50 \
    --timeout 30 \
    -n 10000 \
    -o results.json
```

## Command-Line Options

```
usage: benchmark_mapmatch.py [-h] [--api-url API_URL] [-b BATCH_SIZE]
                             [-t TIMEOUT] [-n NUM_ROUTES] [-o OUTPUT]
                             geojsonl_file

positional arguments:
  geojsonl_file         Input newline-delimited GeoJSON file

optional arguments:
  --api-url API_URL     Map matching API URL (default: http://localhost:8080/match)
  -b, --batch-size      Number of concurrent requests (default: 50)
  -t, --timeout         Request timeout in seconds (default: 30)
  -n, --num-routes      Limit number of routes to process (default: all)
  -o, --output          Output file for detailed results (default: benchmark_results.json)
```

## API Format

Your map matching API should accept POST requests with this format:

**Request:**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "LineString",
        "coordinates": [
          [-77.0365, 38.8977],
          [-77.0350, 38.8980],
          [-77.0330, 38.8985]
        ]
      }
    }
  ]
}
```

**Response:**
Can be any JSON format. The benchmark tool tracks:
- Response time
- Status code (200 = success)
- Error messages if failed

## Example Output

```
======================================================================
Starting Map Matching Benchmark
======================================================================
API URL: http://localhost:8080/match
Total routes: 1000
Batch size: 50
Timeout: 30s
======================================================================

Batch 1/20: 50/1000 routes (5.0%) | Batch time: 2.45s | Avg per route: 49.0ms | Success: 50 | Failed: 0
Batch 2/20: 100/1000 routes (10.0%) | Batch time: 2.38s | Avg per route: 47.6ms | Success: 100 | Failed: 0
...
Batch 20/20: 1000/1000 routes (100.0%) | Batch time: 2.51s | Avg per route: 50.2ms | Success: 1000 | Failed: 0

======================================================================
Benchmark Complete
======================================================================
Total routes: 1000
Successful: 1000 (100.0%)
Failed: 0 (0.0%)
Total time: 48.32s
Throughput: 20.70 routes/sec

Response Time Statistics (successful requests):
  Count: 1000
  Min: 15.2ms
  Max: 85.3ms
  Mean: 48.5ms
  Median: 47.8ms
  P95: 62.1ms
  P99: 71.8ms
======================================================================
```

## JSON Output

The tool saves detailed results to a JSON file:

```json
{
  "config": {
    "api_url": "http://localhost:8080/match",
    "batch_size": 50,
    "timeout": 30
  },
  "summary": {
    "total_routes": 1000,
    "successful": 1000,
    "failed": 0,
    "total_seconds": 48.333,
    "throughput_routes_per_sec": 20.69
  },
  "response_times": {
    "statistics": {
      "min_ms": 15.2,
      "max_ms": 85.3,
      "mean_ms": 48.5,
      "median_ms": 47.8,
      "p95_ms": 62.1,
      "p99_ms": 71.8
    }
  },
  "errors": {}
}
```

## Complete Workflow Example

Here's a complete workflow from generating routes to benchmarking:

```bash
# Step 1: Generate 10,000 synthetic routes with 10m jitter
python generate_synthetic_routes.py district-of-columbia.osm.pbf \
    -n 10000 \
    -o test_routes.geojsonl \
    --jitter-meters 10

# Step 2: Validate the routes
python validate_routes.py test_routes.geojsonl

# Step 3: Analyze jitter characteristics
python analyze_jitter.py test_routes.geojsonl

# Step 4: Start your map matching server (in another terminal)
./your_map_matcher --port 8080

# Step 5: Test with a small sample first
python benchmark_mapmatch.py test_routes.geojsonl -n 100 -b 10

# Step 6: Find optimal batch size with 500 routes
for batch in 10 25 50 100 200; do
    python benchmark_mapmatch.py test_routes.geojsonl \
        -n 500 \
        -b $batch \
        -o batch_${batch}.json
    echo "Batch size $batch complete"
done

# Compare throughput in each batch_*.json file to find the sweet spot

# Step 7: Run full benchmark with optimal batch size
python benchmark_mapmatch.py test_routes.geojsonl \
    --batch-size 50 \
    -o full_benchmark_results.json
```

## Performance Tuning

### Finding Optimal Batch Size

The optimal batch size depends on your system and API. Test different values:

- **Too low** (< 10): Underutilizes network and CPU
- **Just right** (10-100): Maximum throughput, stable response times
- **Too high** (> 200): May cause timeouts, memory issues, or diminishing returns

Watch for:
- Increasing error rates → batch size too high
- Response time P99 spikes → server overloaded
- Flat throughput → you've hit the limit

### Factors Affecting Performance

- **Network latency**: Lower latency = higher throughput
- **API processing time**: Faster API = higher throughput  
- **Server capacity**: More CPU/memory = can handle higher concurrency
- **Client system**: Your machine's network and CPU capacity

## Use Cases

- **Load testing**: Stress test your map matching service
- **Performance benchmarking**: Compare different map matchers
- **Capacity planning**: Determine max throughput for scaling
- **Regression testing**: Ensure changes don't hurt performance
- **Optimization validation**: Test the impact of code changes

## Tips

1. **Start small**: Test with `-n 100` before running full dataset
2. **Monitor server**: Watch CPU/memory on your map matching server
3. **Adjust batch size**: If you see timeouts, reduce batch size
4. **Network stability**: Run on same machine or stable network for consistent results
5. **Multiple runs**: Run 3-5 times and average results for accuracy
6. **Check errors**: Non-zero error count means something's wrong
7. **Compare P95/P99**: These matter more than mean for real-world performance
