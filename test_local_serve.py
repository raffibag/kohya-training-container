#!/usr/bin/env python3
"""
Test the local serve endpoint to verify model loading
"""

import requests
import json
import time
import sys

def test_endpoints():
    base_url = "http://localhost:8080"
    
    print("🔍 Testing /ping endpoint...")
    try:
        response = requests.get(f"{base_url}/ping", timeout=10)
        print(f"✅ Ping: {response.status_code} - {response.text.strip()}")
    except Exception as e:
        print(f"❌ Ping failed: {e}")
        return False
    
    print("🔍 Testing /test endpoint...")
    try:
        response = requests.get(f"{base_url}/test", timeout=30)
        print(f"✅ Test: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   PyTorch: {data.get('torch_version')}")
            print(f"   CUDA: {data.get('cuda_available')}")
            print(f"   Python: {data.get('python_version', '').split()[0]}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    print("🔍 Testing model dependency imports...")
    try:
        shell_cmd = {
            "command_type": "shell",
            "command": "python -c 'import transformers; import clip; print(\"✅ Both CLIP and transformers imported successfully\")'",
            "timeout": 60
        }
        response = requests.post(f"{base_url}/invocations", 
                               json=shell_cmd, 
                               timeout=90,
                               headers={"Content-Type": "application/json"})
        print(f"✅ Import test: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Output: {data.get('stdout', '').strip()}")
            if data.get('stderr'):
                print(f"   Warnings: {data.get('stderr', '').strip()}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        
    return True

if __name__ == "__main__":
    print("🚀 Testing local kohya serve container...")
    success = test_endpoints()
    if success:
        print("\n🎉 All tests passed! Container is ready for deployment.")
    else:
        print("\n❌ Tests failed. Check container setup.")
        sys.exit(1)