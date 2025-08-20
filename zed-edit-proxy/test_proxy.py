#!/usr/bin/env python3
"""
Test script for the Zed Edit Prediction Proxy
"""

import json
import asyncio
import httpx
from typing import Dict, Any

# Test data similar to what Zed would send
TEST_REQUESTS = [
    {
        "name": "Function completion",
        "request": {
            "outline": None,
            "input_events": "typing",
            "input_excerpt": """def calculate_fibonacci(n: int) -> int:
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    # cursor is here""",
            "speculated_output": None,
            "can_collect_data": False,
            "diagnostic_groups": None,
            "git_info": None
        }
    },
    {
        "name": "Variable assignment", 
        "request": {
            "outline": None,
            "input_events": "typing",
            "input_excerpt": """import math

def calculate_circle_area(radius: float) -> float:
    area = """,
            "speculated_output": None,
            "can_collect_data": False,
            "diagnostic_groups": None,
            "git_info": None
        }
    },
    {
        "name": "Class method implementation",
        "request": {
            "outline": None, 
            "input_events": "typing",
            "input_excerpt": """class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value: int):
        # cursor is here - need to implement insert""",
            "speculated_output": None,
            "can_collect_data": True,
            "diagnostic_groups": None,
            "git_info": {
                "head_sha": "abc123",
                "remote_origin_url": "https://github.com/user/repo.git",
                "remote_upstream_url": None
            }
        }
    }
]

async def test_proxy(base_url: str = "http://localhost:8080"):
    """Test the proxy with various requests"""
    
    async with httpx.AsyncClient() as client:
        print(f"Testing Zed Edit Prediction Proxy at {base_url}")
        print("=" * 50)
        
        # Test health endpoint first
        try:
            response = await client.get(f"{base_url}/health")
            health_data = response.json()
            print(f"Health check: {health_data['status']}")
            print(f"OpenAI API connected: {health_data['openai_api_connected']}")
            print(f"Config: {health_data['config']}")
            print()
            
            if not health_data['openai_api_connected']:
                print("WARNING: OpenAI API not connected. Make sure your local model server is running.")
                print()
        except Exception as e:
            print(f"Health check failed: {e}")
            return
        
        # Test prediction requests
        for i, test_case in enumerate(TEST_REQUESTS, 1):
            print(f"Test {i}: {test_case['name']}")
            print("-" * 30)
            
            try:
                response = await client.post(
                    f"{base_url}/predict_edits/v2",
                    json=test_case['request'],
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Success!")
                    print(f"Request ID: {result['request_id']}")
                    print(f"Prediction:")
                    print(f"```")
                    print(result['output_excerpt'])
                    print(f"```")
                    
                    # Test accept endpoint
                    accept_response = await client.post(
                        f"{base_url}/predict_edits/accept",
                        json={"request_id": result['request_id']}
                    )
                    if accept_response.status_code == 200:
                        print(f"✅ Accept endpoint working")
                    else:
                        print(f"❌ Accept endpoint failed: {accept_response.status_code}")
                        
                else:
                    print(f"❌ Failed with status {response.status_code}")
                    print(f"Response: {response.text}")
                    
            except Exception as e:
                print(f"❌ Request failed: {e}")
            
            print()

def main():
    """Run the test"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Zed Edit Prediction Proxy")
    parser.add_argument("--url", default="http://localhost:8080", help="Proxy base URL")
    args = parser.parse_args()
    
    asyncio.run(test_proxy(args.url))

if __name__ == "__main__":
    main()