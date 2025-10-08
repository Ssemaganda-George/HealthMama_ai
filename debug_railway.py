#!/usr/bin/env python3
"""
Debug script to test your Railway deployment
Replace YOUR_RAILWAY_URL with your actual Railway app URL
"""

import requests
import json

# Replace this with your actual Railway URL
RAILWAY_URL = "https://your-app-name.railway.app"

def test_endpoint(url, method="GET", data=None):
    """Test an endpoint and show detailed response"""
    print(f"\n🧪 Testing: {method} {url}")
    print("-" * 50)
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'Not set')}")
        
        # Try to parse as JSON first
        try:
            json_response = response.json()
            print(f"JSON Response: {json.dumps(json_response, indent=2)}")
        except:
            # If not JSON, show first 200 chars of HTML/text
            text = response.text[:200]
            print(f"Non-JSON Response (first 200 chars): {text}")
            if "<!doctype" in text.lower() or "<html" in text.lower():
                print("⚠️  ERROR: Server returned HTML instead of JSON!")
                
    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timed out")
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to server")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

def main():
    print("🚂 Railway Deployment Debug Tool")
    print("=" * 50)
    
    # Test health endpoint
    test_endpoint(f"{RAILWAY_URL}/health")
    
    # Test debug endpoint
    test_endpoint(f"{RAILWAY_URL}/test")
    
    # Test main page
    test_endpoint(f"{RAILWAY_URL}/")
    
    # Test chat endpoint
    test_data = {
        "message": "What is diabetes?",
        "model": "diabetes"
    }
    test_endpoint(f"{RAILWAY_URL}/chat", "POST", test_data)

if __name__ == "__main__":
    print("📝 INSTRUCTIONS:")
    print("1. Edit this file and replace YOUR_RAILWAY_URL with your actual Railway URL")
    print("2. Run: python debug_railway.py")
    print("3. Share the output to help debug the issue")
    print()
    
    url = input("Enter your Railway app URL (or press Enter to use default): ").strip()
    if url:
        RAILWAY_URL = url.rstrip('/')
    
    main()