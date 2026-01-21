#!/usr/bin/env python3
"""
Test script for RESEARCHER API authentication and endpoints.
This script tests the stateless API authentication system.
"""
import requests
import json
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:5001"  # Django API port
API_ENDPOINTS = {
    'ping': f"{API_BASE_URL}/api/v1/ping",
    'get_data_info': f"{API_BASE_URL}/api/v1/get-data-info", 
    'start_client': f"{API_BASE_URL}/api/v1/start-client"
}

# Test credentials - Replace with actual values
API_KEY = "YOUR_API_KEY_HERE"  # Generated API key for testData user
CLIENT_IP = "127.0.0.1"        # Your client IP (must be whitelisted)

def create_headers(api_key=API_KEY, client_ip=CLIENT_IP):
    """Create headers for API requests."""
    return {
        'X-API-Key': api_key,
        'X-Client-IP': client_ip,
        'Content-Type': 'application/json',
        'User-Agent': 'MediNet-Test-Client/1.0'
    }

def test_ping():
    """Test the ping endpoint (health check)."""
    print("\n[SEARCH] Testing /ping endpoint...")
    
    try:
        response = requests.get(
            API_ENDPOINTS['ping'],
            headers=create_headers(),
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'pong':
                print("[OK] Ping test PASSED")
                return True
            else:
                print("[ERROR] Ping test FAILED - Wrong response format")
        else:
            print(f"[ERROR] Ping test FAILED - Status code: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Ping test FAILED - Network error: {e}")
        
    return False

def test_get_data_info():
    """Test the get_data_info endpoint."""
    print("\n[SEARCH] Testing /get-data-info endpoint...")
    
    try:
        response = requests.get(
            API_ENDPOINTS['get_data_info'],
            headers=create_headers(),
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response keys: {list(data.keys())}")
            
            # Check if we have dataset structure
            if 'dataset_id' in data:
                dataset_count = len(data['dataset_id'])
                print(f"[OK] Found {dataset_count} datasets accessible to user")
                
                # Print first dataset info if available
                if dataset_count > 0:
                    print("[INFO] First dataset:")
                    for key in data.keys():
                        if data[key]:  # If list is not empty
                            print(f"  {key}: {data[key][0]}")
                return True
            else:
                print(f"[ERROR] Unexpected response format: {data}")
        else:
            error_data = response.json() if response.content else {}
            print(f"[ERROR] Get data info test FAILED: {error_data}")
            
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Get data info test FAILED - Network error: {e}")
    except json.JSONDecodeError as e:
        print(f"[ERROR] Get data info test FAILED - JSON decode error: {e}")
        
    return False

def test_start_client():
    """Test the start_client endpoint with sample model configuration."""
    print("\n[SEARCH] Testing /start-client endpoint...")
    
    # Sample model configuration
    model_config = {
        "framework": "pt",
        "model": {
            "dataset": {
                "selected_datasets": [
                    {
                        "dataset_id": 1,  # Replace with actual dataset ID
                        "dataset_name": "test_dataset"
                    }
                ]
            },
            "architecture": {
                "layers": [
                    {
                        "type": "linear",
                        "params": {
                            "in_features": 4,
                            "out_features": 10
                        }
                    },
                    {
                        "type": "linear", 
                        "params": {
                            "in_features": 10,
                            "out_features": 3
                        }
                    }
                ]
            }
        }
    }
    
    payload = {
        "model_json": model_config,
        "server_address": "localhost:8080",
        "client_id": f"test_client_{int(time.time())}"
    }
    
    try:
        response = requests.post(
            API_ENDPOINTS['start_client'],
            headers=create_headers(),
            json=payload,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data}")
            
            if data.get('status') == 'Flower Client started':
                print("[OK] Start client test PASSED")
                return True
            else:
                print("[ERROR] Start client test FAILED - Wrong response format")
        else:
            error_data = response.json() if response.content else {}
            print(f"[ERROR] Start client test FAILED: {error_data}")
            
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Start client test FAILED - Network error: {e}")
    except json.JSONDecodeError as e:
        print(f"[ERROR] Start client test FAILED - JSON decode error: {e}")
        
    return False

def test_invalid_api_key():
    """Test with invalid API key to verify security."""
    print("\n[SEARCH] Testing invalid API key security...")
    
    try:
        response = requests.get(
            API_ENDPOINTS['ping'],
            headers=create_headers(api_key="invalid_key_123"),
            timeout=10
        )
        
        if response.status_code == 401:
            print("[OK] Security test PASSED - Invalid API key properly rejected")
            return True
        else:
            print(f"[ERROR] Security test FAILED - Should return 401, got {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Security test FAILED - Network error: {e}")
        
    return False

def main():
    """Run all API tests."""
    print("🚀 Starting MediNet API Tests")
    print(f"[INFO] Base URL: {API_BASE_URL}")
    print(f"🔑 API Key: {API_KEY[:10]}..." if len(API_KEY) > 10 else "[ERROR] API_KEY_NOT_SET")
    print(f"🌐 Client IP: {CLIENT_IP}")
    print("=" * 60)
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("[ERROR] Please set your actual API key in the script!")
        print("\nTo generate API key, run:")
        print("python manage.py generate_api_key testData")
        return
    
    # Run all tests
    results = {
        'ping': test_ping(),
        'invalid_key_security': test_invalid_api_key(),
        'get_data_info': test_get_data_info(),
        'start_client': test_start_client()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("[INFO] TEST RESULTS SUMMARY:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "[OK] PASSED" if passed_test else "[ERROR] FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED! API is working correctly.")
    else:
        print("[WARNING]  Some tests FAILED. Check the logs above for details.")

if __name__ == "__main__":
    main()