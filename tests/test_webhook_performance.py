"""
Webhook Performance Testing Script

This script tests the performance of your webhook endpoint to ensure
it meets the sub-100ms response time target.
"""

import asyncio
import aiohttp
import time
import statistics
import json
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class PerformanceResult:
    """Performance test result."""

    response_time: float
    status_code: int
    success: bool
    error: str = None


class WebhookPerformanceTester:
    """Performance tester for webhook endpoints."""

    def __init__(self, webhook_url: str, token: str):
        self.webhook_url = f"{webhook_url}/webhook/{token}"
        self.session = None

    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=50,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=5.0, connect=1.0)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def create_test_update(self, update_id: int) -> Dict[str, Any]:
        """Create a test Telegram update."""
        return {
            "update_id": update_id,
            "message": {
                "message_id": update_id,
                "date": int(time.time()),
                "chat": {"id": 12345, "type": "private"},
                "from": {
                    "id": 67890,
                    "is_bot": False,
                    "first_name": "Test",
                    "username": "testuser",
                },
                "text": "Hello, bot!",
            },
        }

    async def send_single_request(self, update_id: int) -> PerformanceResult:
        """Send a single webhook request and measure performance."""
        update_data = self.create_test_update(update_id)

        start_time = time.time()
        try:
            async with self.session.post(
                self.webhook_url, json=update_data
            ) as response:
                await response.text()  # Read response body
                end_time = time.time()

                return PerformanceResult(
                    response_time=(end_time - start_time) * 1000,  # Convert to ms
                    status_code=response.status,
                    success=200 <= response.status < 300,
                )

        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            # Provide more helpful error messages
            if "Connection refused" in error_msg or "ConnectError" in error_msg:
                error_msg = "Connection refused - is the webhook server running?"
            elif "timeout" in error_msg.lower():
                error_msg = "Request timeout - server may be overloaded"

            return PerformanceResult(
                response_time=(end_time - start_time) * 1000,
                status_code=0,
                success=False,
                error=error_msg,
            )

    async def run_concurrent_test(
        self, num_requests: int, concurrency: int = 10
    ) -> List[PerformanceResult]:
        """Run concurrent performance test with proper pacing."""
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(update_id: int) -> PerformanceResult:
            async with semaphore:
                result = await self.send_single_request(update_id)
                # Add small delay between requests to simulate real-world usage
                await asyncio.sleep(0.1)  # 100ms between requests
                return result

        tasks = [bounded_request(i + 1) for i in range(num_requests)]

        return await asyncio.gather(*tasks)

    async def run_load_test(
        self, duration_seconds: int = 60, requests_per_second: int = 10
    ) -> List[PerformanceResult]:
        """Run sustained load test."""
        results = []
        start_time = time.time()
        request_id = 0

        while time.time() - start_time < duration_seconds:
            batch_start = time.time()

            # Send batch of requests
            batch_tasks = []
            for _ in range(requests_per_second):
                request_id += 1
                batch_tasks.append(self.send_single_request(request_id))

            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)

            # Wait for next second
            batch_duration = time.time() - batch_start
            sleep_time = max(0, 1.0 - batch_duration)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        return results


def analyze_results(results: List[PerformanceResult]) -> Dict[str, Any]:
    """Analyze performance test results."""
    if not results:
        return {"error": "No results to analyze"}

    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]

    if successful_results:
        response_times = [r.response_time for r in successful_results]

        analysis = {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "success_rate": len(successful_results) / len(results) * 100,
            "response_times": {
                "min": min(response_times),
                "max": max(response_times),
                "mean": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "p95": (
                    statistics.quantiles(response_times, n=20)[18]
                    if len(response_times) > 20
                    else max(response_times)
                ),
                "p99": (
                    statistics.quantiles(response_times, n=100)[98]
                    if len(response_times) > 100
                    else max(response_times)
                ),
            },
            "performance_targets": {
                "sub_100ms": sum(1 for rt in response_times if rt < 100)
                / len(response_times)
                * 100,
                "sub_50ms": sum(1 for rt in response_times if rt < 50)
                / len(response_times)
                * 100,
                "sub_20ms": sum(1 for rt in response_times if rt < 20)
                / len(response_times)
                * 100,
            },
        }

        # Add error analysis
        if failed_results:
            error_types = {}
            for result in failed_results:
                error_type = result.error or f"HTTP_{result.status_code}"
                error_types[error_type] = error_types.get(error_type, 0) + 1
            analysis["errors"] = error_types

        return analysis

    else:
        return {
            "total_requests": len(results),
            "successful_requests": 0,
            "failed_requests": len(failed_results),
            "success_rate": 0,
            "error": "All requests failed",
        }


def print_performance_report(analysis: Dict[str, Any]) -> None:
    """Print formatted performance report."""
    print("=" * 60)
    print("WEBHOOK PERFORMANCE TEST RESULTS")
    print("=" * 60)

    if "error" in analysis:
        print(f"❌ Error: {analysis['error']}")
        return

    # Overall stats
    print(f"📊 Total Requests: {analysis['total_requests']}")
    print(f"✅ Successful: {analysis['successful_requests']}")
    print(f"❌ Failed: {analysis['failed_requests']}")
    print(f"📈 Success Rate: {analysis['success_rate']:.1f}%")
    print()

    # Response time stats
    if "response_times" in analysis:
        rt = analysis["response_times"]
        print("⏱️  RESPONSE TIME STATISTICS (ms)")
        print("-" * 40)
        print(f"Min:      {rt['min']:.1f}")
        print(f"Max:      {rt['max']:.1f}")
        print(f"Mean:     {rt['mean']:.1f}")
        print(f"Median:   {rt['median']:.1f}")
        print(f"P95:      {rt['p95']:.1f}")
        print(f"P99:      {rt['p99']:.1f}")
        print()

        # Performance targets
        targets = analysis["performance_targets"]
        print("🎯 PERFORMANCE TARGETS")
        print("-" * 40)
        print(
            f"< 20ms:   {targets['sub_20ms']:.1f}% {'✅' if targets['sub_20ms'] > 80 else '⚠️'}"
        )
        print(
            f"< 50ms:   {targets['sub_50ms']:.1f}% {'✅' if targets['sub_50ms'] > 90 else '⚠️'}"
        )
        print(
            f"< 100ms:  {targets['sub_100ms']:.1f}% {'✅' if targets['sub_100ms'] > 95 else '⚠️'}"
        )
        print()

    # Error analysis
    if "errors" in analysis:
        print("🚨 ERROR BREAKDOWN")
        print("-" * 40)
        for error_type, count in analysis["errors"].items():
            print(f"{error_type}: {count}")
        print()

    # Performance verdict
    if "response_times" in analysis:
        mean_time = analysis["response_times"]["mean"]
        sub_100_rate = analysis["performance_targets"]["sub_100ms"]

        if mean_time < 50 and sub_100_rate > 95:
            print("🏆 VERDICT: EXCELLENT PERFORMANCE!")
            print("   Your webhook is highly optimized.")
        elif mean_time < 100 and sub_100_rate > 90:
            print("✅ VERDICT: GOOD PERFORMANCE")
            print("   Your webhook meets performance targets.")
        elif mean_time < 200:
            print("⚠️  VERDICT: ACCEPTABLE PERFORMANCE")
            print("   Consider optimizations for better response times.")
        else:
            print("❌ VERDICT: POOR PERFORMANCE")
            print("   Significant optimizations needed.")


async def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test webhook performance")
    parser.add_argument(
        "--url", required=True, help="Webhook base URL (e.g., http://localhost:8000)"
    )
    parser.add_argument("--token", required=True, help="Bot token")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Concurrent requests"
    )
    parser.add_argument(
        "--load-test", action="store_true", help="Run sustained load test"
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="Load test duration (seconds)"
    )
    parser.add_argument(
        "--rps", type=int, default=10, help="Requests per second for load test"
    )

    args = parser.parse_args()

    async with WebhookPerformanceTester(args.url, args.token) as tester:
        print(f"🚀 Starting performance test...")
        print(f"   URL: {args.url}/webhook/{args.token}")

        if args.load_test:
            print(f"   Mode: Load test ({args.duration}s at {args.rps} RPS)")
            results = await tester.run_load_test(args.duration, args.rps)
        else:
            print(
                f"   Mode: Burst test ({args.requests} requests, {args.concurrency} concurrent)"
            )
            results = await tester.run_concurrent_test(args.requests, args.concurrency)

        analysis = analyze_results(results)
        print_performance_report(analysis)


if __name__ == "__main__":
    asyncio.run(main())
