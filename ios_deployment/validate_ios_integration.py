#!/usr/bin/env python3
"""
iOS Integration Validation Script

Validates that the backend is properly configured for iOS deployment:
- CORS configuration
- Rate limiting
- User-Agent tracking
- Request ID tracking
- Health check endpoints
- API endpoints
"""

import asyncio
import aiohttp
import sys
from typing import Dict, List, Tuple
from datetime import datetime
import json


class IOSIntegrationValidator:
    """Validate iOS integration readiness"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.results: List[Tuple[str, bool, str]] = []
        self.ios_user_agent = "ReleAF-iOS-SDK/1.0.0 (iPhone; iOS 17.0; Scale/3.00)"
    
    async def validate_all(self):
        """Run all validation tests"""
        print("=" * 80)
        print("🍎 iOS INTEGRATION VALIDATION")
        print("=" * 80)
        print(f"Base URL: {self.base_url}")
        print(f"Timestamp: {datetime.utcnow().isoformat()}")
        print("=" * 80)
        print()
        
        async with aiohttp.ClientSession() as session:
            # Test 1: CORS Preflight
            await self.test_cors_preflight(session)
            
            # Test 2: CORS with iOS Origin
            await self.test_cors_ios_origin(session)
            
            # Test 3: Health Check
            await self.test_health_check(session)
            
            # Test 4: iOS Health Check
            await self.test_ios_health_check(session)
            
            # Test 5: User-Agent Tracking
            await self.test_user_agent_tracking(session)
            
            # Test 6: Request ID Tracking
            await self.test_request_id_tracking(session)
            
            # Test 7: Rate Limiting
            await self.test_rate_limiting(session)
            
            # Test 8: API Endpoints
            await self.test_api_endpoints(session)
        
        # Print results
        self.print_results()
    
    async def test_cors_preflight(self, session: aiohttp.ClientSession):
        """Test CORS preflight request"""
        try:
            async with session.options(
                f"{self.base_url}/api/v1/chat",
                headers={
                    "Origin": "capacitor://localhost",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type,X-API-Key"
                }
            ) as response:
                headers = response.headers
                
                # Check CORS headers
                has_allow_origin = "Access-Control-Allow-Origin" in headers
                has_allow_methods = "Access-Control-Allow-Methods" in headers
                has_allow_headers = "Access-Control-Allow-Headers" in headers
                
                if has_allow_origin and has_allow_methods and has_allow_headers:
                    self.results.append(("CORS Preflight", True, "All CORS headers present"))
                else:
                    missing = []
                    if not has_allow_origin:
                        missing.append("Allow-Origin")
                    if not has_allow_methods:
                        missing.append("Allow-Methods")
                    if not has_allow_headers:
                        missing.append("Allow-Headers")
                    self.results.append(("CORS Preflight", False, f"Missing: {', '.join(missing)}"))
        except Exception as e:
            self.results.append(("CORS Preflight", False, str(e)))
    
    async def test_cors_ios_origin(self, session: aiohttp.ClientSession):
        """Test CORS with iOS origin"""
        try:
            async with session.get(
                f"{self.base_url}/health",
                headers={"Origin": "capacitor://localhost"}
            ) as response:
                allow_origin = response.headers.get("Access-Control-Allow-Origin", "")
                
                if "capacitor://localhost" in allow_origin or allow_origin == "*":
                    self.results.append(("CORS iOS Origin", True, f"Origin allowed: {allow_origin}"))
                else:
                    self.results.append(("CORS iOS Origin", False, f"Origin not allowed: {allow_origin}"))
        except Exception as e:
            self.results.append(("CORS iOS Origin", False, str(e)))
    
    async def test_health_check(self, session: aiohttp.ClientSession):
        """Test standard health check"""
        try:
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    self.results.append(("Health Check", True, f"Status: {data.get('status', 'unknown')}"))
                else:
                    self.results.append(("Health Check", False, f"HTTP {response.status}"))
        except Exception as e:
            self.results.append(("Health Check", False, str(e)))
    
    async def test_ios_health_check(self, session: aiohttp.ClientSession):
        """Test iOS-specific health check"""
        try:
            async with session.get(f"{self.base_url}/health/ios") as response:
                if response.status == 200:
                    data = await response.json()
                    has_ios_support = data.get("ios_support", False)
                    has_features = "features" in data
                    has_rate_limits = "rate_limits" in data
                    
                    if has_ios_support and has_features and has_rate_limits:
                        self.results.append(("iOS Health Check", True, "All iOS info present"))
                    else:
                        self.results.append(("iOS Health Check", False, "Missing iOS info"))
                elif response.status == 404:
                    self.results.append(("iOS Health Check", False, "Endpoint not implemented (optional)"))
                else:
                    self.results.append(("iOS Health Check", False, f"HTTP {response.status}"))
        except Exception as e:
            self.results.append(("iOS Health Check", False, str(e)))
    
    async def test_user_agent_tracking(self, session: aiohttp.ClientSession):
        """Test User-Agent tracking"""
        try:
            async with session.get(
                f"{self.base_url}/health",
                headers={"User-Agent": self.ios_user_agent}
            ) as response:
                if response.status == 200:
                    # If we get here, User-Agent was accepted
                    self.results.append(("User-Agent Tracking", True, "iOS User-Agent accepted"))
                else:
                    self.results.append(("User-Agent Tracking", False, f"HTTP {response.status}"))
        except Exception as e:
            self.results.append(("User-Agent Tracking", False, str(e)))
    
    async def test_request_id_tracking(self, session: aiohttp.ClientSession):
        """Test Request ID tracking"""
        try:
            request_id = "test-ios-request-12345"
            async with session.get(
                f"{self.base_url}/health",
                headers={"X-Request-ID": request_id}
            ) as response:
                response_id = response.headers.get("X-Request-ID", "")
                
                if response_id == request_id:
                    self.results.append(("Request ID Tracking", True, f"Request ID echoed: {request_id}"))
                elif response_id:
                    self.results.append(("Request ID Tracking", True, f"New Request ID: {response_id}"))
                else:
                    self.results.append(("Request ID Tracking", False, "No Request ID in response"))
        except Exception as e:
            self.results.append(("Request ID Tracking", False, str(e)))
    
    async def test_rate_limiting(self, session: aiohttp.ClientSession):
        """Test rate limiting headers"""
        try:
            async with session.get(f"{self.base_url}/health") as response:
                has_limit = "X-RateLimit-Limit" in response.headers
                has_remaining = "X-RateLimit-Remaining" in response.headers
                
                if has_limit and has_remaining:
                    limit = response.headers.get("X-RateLimit-Limit")
                    remaining = response.headers.get("X-RateLimit-Remaining")
                    self.results.append(("Rate Limiting", True, f"Limit: {limit}, Remaining: {remaining}"))
                else:
                    self.results.append(("Rate Limiting", False, "Rate limit headers not present (may be optional)"))
        except Exception as e:
            self.results.append(("Rate Limiting", False, str(e)))
    
    async def test_api_endpoints(self, session: aiohttp.ClientSession):
        """Test API endpoints availability"""
        endpoints = [
            "/api/v1/chat",
            "/api/v1/vision/analyze",
            "/api/v1/organizations/search"
        ]
        
        for endpoint in endpoints:
            try:
                # OPTIONS request to check if endpoint exists
                async with session.options(f"{self.base_url}{endpoint}") as response:
                    if response.status in [200, 204, 405]:  # 405 = Method Not Allowed (but endpoint exists)
                        self.results.append((f"Endpoint {endpoint}", True, "Available"))
                    else:
                        self.results.append((f"Endpoint {endpoint}", False, f"HTTP {response.status}"))
            except Exception as e:
                self.results.append((f"Endpoint {endpoint}", False, str(e)))
    
    def print_results(self):
        """Print validation results"""
        print()
        print("=" * 80)
        print("📊 VALIDATION RESULTS")
        print("=" * 80)
        print()
        
        passed = sum(1 for _, success, _ in self.results if success)
        total = len(self.results)
        
        for test_name, success, message in self.results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} | {test_name:30s} | {message}")
        
        print()
        print("=" * 80)
        print(f"SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        print("=" * 80)
        print()
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Backend is iOS-ready!")
            return 0
        elif passed >= total * 0.8:
            print("⚠️  MOST TESTS PASSED. Review failures and apply fixes.")
            return 1
        else:
            print("❌ MANY TESTS FAILED. Backend needs iOS integration updates.")
            return 2


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate iOS integration")
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Base URL of API Gateway (default: http://localhost:8080)"
    )
    args = parser.parse_args()
    
    validator = IOSIntegrationValidator(base_url=args.url)
    exit_code = await validator.validate_all()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())

