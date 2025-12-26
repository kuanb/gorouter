#!/usr/bin/env python3
"""
Benchmark a local map matching API with synthetic routes.

Sends routes from a newline-delimited GeoJSON file to a map matching API
with configurable concurrency for performance testing.
"""

import json
import argparse
import time
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from collections import defaultdict


class MapMatchBenchmark:
    def __init__(self, api_url: str, batch_size: int = 50, timeout: int = 30):
        self.api_url = api_url
        self.batch_size = batch_size
        self.timeout = timeout
        self.results = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'errors': defaultdict(int),
            'response_times': [],
            'start_time': None,
            'end_time': None
        }
    
    async def match_route(self, session: aiohttp.ClientSession, 
                         route_feature: Dict[str, Any], 
                         route_id: int) -> Dict[str, Any]:
        """Send a single route to the map matching API."""
        # Wrap the feature in a FeatureCollection as expected by the API
        payload = {
            "type": "FeatureCollection",
            "features": [route_feature]
        }
        
        start_time = time.time()
        result = {
            'route_id': route_id,
            'success': False,
            'error': None,
            'response_time': None,
            'status_code': None
        }
        
        try:
            async with session.post(
                self.api_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response_time = time.time() - start_time
                result['response_time'] = response_time
                result['status_code'] = response.status
                
                if response.status == 200:
                    # Optionally parse the response
                    response_data = await response.json()
                    result['success'] = True
                    result['response_data'] = response_data
                else:
                    result['error'] = f"HTTP {response.status}"
                    error_text = await response.text()
                    result['error_detail'] = error_text[:200]  # First 200 chars
                    
        except asyncio.TimeoutError:
            result['error'] = 'Timeout'
            result['response_time'] = self.timeout
        except aiohttp.ClientError as e:
            result['error'] = f'ClientError: {type(e).__name__}'
            result['response_time'] = time.time() - start_time
        except Exception as e:
            result['error'] = f'Exception: {type(e).__name__}: {str(e)}'
            result['response_time'] = time.time() - start_time
        
        return result
    
    async def process_batch(self, session: aiohttp.ClientSession, 
                           routes: List[tuple]) -> List[Dict[str, Any]]:
        """Process a batch of routes concurrently."""
        tasks = [
            self.match_route(session, route_feature, route_id)
            for route_id, route_feature in routes
        ]
        return await asyncio.gather(*tasks)
    
    async def run_benchmark(self, routes: List[Dict[str, Any]]):
        """Run the full benchmark with batched concurrent requests."""
        total_routes = len(routes)
        self.results['total'] = total_routes
        self.results['start_time'] = datetime.now()
        
        print(f"\n{'='*70}")
        print(f"Starting Map Matching Benchmark")
        print(f"{'='*70}")
        print(f"API URL: {self.api_url}")
        print(f"Total routes: {total_routes}")
        print(f"Batch size: {self.batch_size}")
        print(f"Timeout: {self.timeout}s")
        print(f"{'='*70}\n")
        
        # Create route batches
        batches = []
        for i in range(0, total_routes, self.batch_size):
            batch = [(j, routes[j]) for j in range(i, min(i + self.batch_size, total_routes))]
            batches.append(batch)
        
        # Process batches
        async with aiohttp.ClientSession() as session:
            for batch_idx, batch in enumerate(batches, 1):
                batch_start = time.time()
                
                results = await self.process_batch(session, batch)
                
                batch_time = time.time() - batch_start
                
                # Update statistics
                for result in results:
                    if result['success']:
                        self.results['success'] += 1
                        self.results['response_times'].append(result['response_time'])
                    else:
                        self.results['failed'] += 1
                        self.results['errors'][result['error']] += 1
                
                # Progress update
                processed = min(batch_idx * self.batch_size, total_routes)
                percent = (processed / total_routes) * 100
                avg_time = batch_time / len(batch)
                
                print(f"Batch {batch_idx}/{len(batches)}: "
                      f"{processed}/{total_routes} routes ({percent:.1f}%) | "
                      f"Batch time: {batch_time:.2f}s | "
                      f"Avg per route: {avg_time*1000:.1f}ms | "
                      f"Success: {self.results['success']} | "
                      f"Failed: {self.results['failed']}")
        
        self.results['end_time'] = datetime.now()
    
    def print_summary(self):
        """Print benchmark summary statistics."""
        total_time = (self.results['end_time'] - self.results['start_time']).total_seconds()
        
        print(f"\n{'='*70}")
        print(f"Benchmark Complete")
        print(f"{'='*70}")
        print(f"Total routes: {self.results['total']}")
        print(f"Successful: {self.results['success']} ({self.results['success']/self.results['total']*100:.1f}%)")
        print(f"Failed: {self.results['failed']} ({self.results['failed']/self.results['total']*100:.1f}%)")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {self.results['total']/total_time:.2f} routes/sec")
        
        if self.results['response_times']:
            times = self.results['response_times']
            times_ms = [t * 1000 for t in times]
            times_sorted = sorted(times_ms)
            
            print(f"\nResponse Time Statistics (successful requests):")
            print(f"  Count: {len(times)}")
            print(f"  Min: {min(times_ms):.1f}ms")
            print(f"  Max: {max(times_ms):.1f}ms")
            print(f"  Mean: {sum(times_ms)/len(times_ms):.1f}ms")
            print(f"  Median: {times_sorted[len(times_sorted)//2]:.1f}ms")
            print(f"  P95: {times_sorted[int(len(times_sorted)*0.95)]:.1f}ms")
            print(f"  P99: {times_sorted[int(len(times_sorted)*0.99)]:.1f}ms")
        
        if self.results['errors']:
            print(f"\nError Breakdown:")
            for error, count in sorted(self.results['errors'].items(), 
                                      key=lambda x: x[1], reverse=True):
                print(f"  {error}: {count}")
        
        print(f"{'='*70}\n")
    
    def save_results(self, output_file: str):
        """Save detailed results to a JSON file."""
        output = {
            'config': {
                'api_url': self.api_url,
                'batch_size': self.batch_size,
                'timeout': self.timeout
            },
            'summary': {
                'total_routes': self.results['total'],
                'successful': self.results['success'],
                'failed': self.results['failed'],
                'start_time': self.results['start_time'].isoformat(),
                'end_time': self.results['end_time'].isoformat(),
                'total_seconds': (self.results['end_time'] - self.results['start_time']).total_seconds(),
                'throughput_routes_per_sec': self.results['total'] / (self.results['end_time'] - self.results['start_time']).total_seconds()
            },
            'response_times': {
                'values_ms': [t * 1000 for t in self.results['response_times']],
                'count': len(self.results['response_times'])
            },
            'errors': dict(self.results['errors'])
        }
        
        if self.results['response_times']:
            times_ms = [t * 1000 for t in self.results['response_times']]
            times_sorted = sorted(times_ms)
            output['response_times']['statistics'] = {
                'min_ms': min(times_ms),
                'max_ms': max(times_ms),
                'mean_ms': sum(times_ms) / len(times_ms),
                'median_ms': times_sorted[len(times_sorted)//2],
                'p95_ms': times_sorted[int(len(times_sorted)*0.95)],
                'p99_ms': times_sorted[int(len(times_sorted)*0.99)]
            }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Detailed results saved to: {output_file}")


def load_routes(geojsonl_file: str) -> List[Dict[str, Any]]:
    """Load routes from newline-delimited GeoJSON file."""
    routes = []
    with open(geojsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                feature = json.loads(line)
                routes.append(feature)
    return routes


async def main():
    parser = argparse.ArgumentParser(
        description='Benchmark map matching API with synthetic routes'
    )
    parser.add_argument('geojsonl_file', help='Input newline-delimited GeoJSON file')
    parser.add_argument('--api-url', default='http://localhost:8080/match',
                       help='Map matching API URL (default: http://localhost:8080/match)')
    parser.add_argument('-b', '--batch-size', type=int, default=50,
                       help='Number of concurrent requests (default: 50)')
    parser.add_argument('-t', '--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('-n', '--num-routes', type=int, default=None,
                       help='Limit number of routes to process (default: all)')
    parser.add_argument('-o', '--output', default='benchmark_results.json',
                       help='Output file for detailed results (default: benchmark_results.json)')
    
    args = parser.parse_args()
    
    # Load routes
    print(f"Loading routes from {args.geojsonl_file}...")
    routes = load_routes(args.geojsonl_file)
    print(f"Loaded {len(routes)} routes")
    
    # Limit if requested
    if args.num_routes and args.num_routes < len(routes):
        routes = routes[:args.num_routes]
        print(f"Limited to first {args.num_routes} routes")
    
    # Run benchmark
    benchmark = MapMatchBenchmark(
        api_url=args.api_url,
        batch_size=args.batch_size,
        timeout=args.timeout
    )
    
    await benchmark.run_benchmark(routes)
    
    # Print summary
    benchmark.print_summary()
    
    # Save results
    benchmark.save_results(args.output)


if __name__ == '__main__':
    asyncio.run(main())
